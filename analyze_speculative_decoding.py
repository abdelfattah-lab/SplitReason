import os
import time
import csv
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    LogitsProcessor,
    TextStreamer
)
from datasets import load_dataset
from typing import List, Dict, Tuple, Optional, Union, Any

# ==========================================================
# Configuration
# ==========================================================

class Config:
    # Model paths
    small_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    large_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    
    # Speculative decoding parameters
    gamma = 1  # Acceptance threshold
    max_new_tokens = 16384  # Maximum number of tokens to generate
    max_length = 16384  # Maximum context length
    spec_len = 32  # Number of tokens to speculate at once
    
    # Dataset
    dataset_name = "simplescaling/aime24_nofigures"  # AIME dataset from HuggingFace
    
    # Prompt configuration
    terminating_string = " \n Put your final answer within \\boxed{}."
    model_think_prefix = "<think>\n"
    prompt_template = "<｜begin▁of▁sentence｜><｜User｜>{problem}{terminating_string}<｜Assistant｜>\n{model_think_prefix}"
    
    # Output
    output_dir = "speculative_decoding_results"
    csv_filename = "token_analysis.csv"
    
    # Visualization
    visualization_dir = "visualizations"


# ==========================================================
# Custom Dataset for AIME Problems
# ==========================================================

class AIMEDataset:
    """Custom dataset loader for AIME mathematics problems."""
    
    def __init__(self, config: Config):
        """Initialize the dataset."""
        self.config = config
        
        # Load AIME problems
        self.problems = self._load_aime_problems()
    
    def _load_aime_problems(self) -> List[Dict[str, str]]:
        """
        Load AIME problems from the specified dataset.
        
        Returns:
            List of dictionaries containing problem ID and text.
        """
        # Load the dataset from HuggingFace
        try:
            print(f"Loading dataset: {self.config.dataset_name}")
            dataset = load_dataset(self.config.dataset_name)
            
            # Process the dataset
            problems = []
            for idx, row in enumerate(dataset["train"]):
                # Extract question and ID
                question = row.get('problem', row.get('question', ''))
                question_id = row.get('id', f'question_{idx}')
                
                # Skip if question is empty
                if not question:
                    print(f"Skipping empty question at index {idx}")
                    continue
                
                problems.append({
                    "id": question_id,
                    "text": question
                })
            
            print(f"Successfully loaded {len(problems)} problems from {self.config.dataset_name}")
            return problems
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            
            raise e
    
    def __len__(self) -> int:
        """Return the number of problems in the dataset."""
        return len(self.problems)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        """Get a problem by index."""
        return self.problems[idx]


# ==========================================================
# Speculative Decoding Implementation
# ==========================================================

class SpeculativeDecoder:
    """Implements speculative decoding between a small and large language model."""
    
    def __init__(self, config: Config):
        """Initialize the speculative decoder with small and large models."""
        self.config = config
        
        print(f"Loading small model: {config.small_model_name}")
        self.small_model = AutoModelForCausalLM.from_pretrained(
            config.small_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.small_tokenizer = AutoTokenizer.from_pretrained(
            config.small_model_name,
            trust_remote_code=True
        )
        
        print(f"Loading large model: {config.large_model_name}")
        self.large_model = AutoModelForCausalLM.from_pretrained(
            config.large_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.large_tokenizer = AutoTokenizer.from_pretrained(
            config.large_model_name,
            trust_remote_code=True
        )
        
        # Initialize token tracking
        self._init_token_tracking()
    
    def _init_token_tracking(self):
        """Initialize token tracking for analysis."""
        self.token_records = []
        self.current_token_id = 0
    
    def _record_token(self, token_text: str, source: str, prob_small: float, 
                     prob_big: Optional[float] = None, 
                     rejection_reason: Optional[str] = None):
        """
        Record a token's information for later analysis.
        
        Args:
            token_text: The decoded token text
            source: Either "small_model" or "big_model"
            prob_small: Probability assigned by the small model
            prob_big: Probability assigned by the large model (if available)
            rejection_reason: Reason for rejection (if rejected)
        """
        self.token_records.append({
            "token_id": self.current_token_id,
            "token_text": token_text,
            "source": source,
            "probability_small": prob_small,
            "probability_big": prob_big,
            "rejection_reason": rejection_reason
        })
        self.current_token_id += 1
    
    def speculative_decode(self, prompt: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Perform speculative decoding with the given prompt.
        
        Args:
            prompt: The input prompt to generate from
            
        Returns:
            Tuple of (generated text, token records)
        """
        # Reset token tracking
        self._init_token_tracking()
        
        # Tokenize prompt
        input_ids = self.small_tokenizer(prompt, return_tensors="pt").input_ids.to(self.small_model.device)
        prompt_len = input_ids.shape[1]
        
        # Initialize generation
        generated_ids = input_ids.clone()
        
        # Generate until we reach max length or end of sequence
        with torch.no_grad():
            while generated_ids.shape[1] - prompt_len < self.config.max_new_tokens:
                # Get current context (limited by max length)
                curr_context = generated_ids[:, -self.config.max_length:]
                
                # Step 1: Draft tokens with small model
                draft_ids, draft_probs = self._generate_draft_tokens(curr_context)
                
                if len(draft_ids) == 0:
                    break  # No more tokens to generate
                
                # Step 2: Verify tokens with large model
                accepted_ids, big_model_probs, accepted_mask = self._verify_draft_tokens(
                    curr_context, draft_ids, draft_probs
                )
                
                # Step 3: Record token information for analysis
                self._record_token_info(draft_ids, accepted_ids, accepted_mask, draft_probs, big_model_probs)
                
                # Step 4: Append accepted tokens to generated sequence
                if len(accepted_ids) > 0:
                    generated_ids = torch.cat([generated_ids, accepted_ids.unsqueeze(0)], dim=1)
                
                # Step 5: If any token was rejected or we didn't accept all tokens,
                # generate one token from large model
                if not all(accepted_mask) or len(accepted_ids) < len(draft_ids):
                    big_token, big_prob = self._generate_single_token(generated_ids)
                    
                    if big_token is None:
                        break  # End of sequence
                    
                    # Record the big model's token
                    self._record_token(
                        token_text=self.large_tokenizer.decode([big_token]),
                        source="big_model",
                        prob_small=0.0,  # Small model didn't generate this # TODO: Better to actually calculate the probability of the small model generating this token.
                        prob_big=big_prob
                    )
                    
                    # Append the large model's token
                    generated_ids = torch.cat([generated_ids, torch.tensor([[big_token]]).to(generated_ids.device)], dim=1)
                
                # Check if we've hit the end token
                if generated_ids[0, -1].item() == self.small_tokenizer.eos_token_id:
                    break
        
        # Decode the full generated sequence
        generated_text = self.small_tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)
        
        return generated_text, self.token_records
    
    def _generate_draft_tokens(self, input_ids: torch.Tensor) -> Tuple[List[int], List[float]]:
        """
        Generate draft tokens using the small model.
        
        Args:
            input_ids: Current context tokens
            
        Returns:
            Tuple of (draft token ids, probabilities)
        """
        # Get the output from the small model
        with torch.no_grad():
            outputs = self.small_model(input_ids)
            
        logits = outputs.logits[:, -1, :]  # Get logits for the last token
        probs = torch.softmax(logits, dim=-1)
        
        # Sample tokens from the small model
        draft_ids = []
        draft_probs = []
        
        curr_input_ids = input_ids.clone()
        
        # Generate up to spec_len tokens
        for _ in range(self.config.spec_len):
            logits = self.small_model(curr_input_ids).logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            
            # Sample a token
            next_token = torch.multinomial(probs[0], 1).item()
            next_token_prob = probs[0, next_token].item()
            
            draft_ids.append(next_token)
            draft_probs.append(next_token_prob)
            
            # Update context with the new token
            curr_input_ids = torch.cat([curr_input_ids, torch.tensor([[next_token]]).to(curr_input_ids.device)], dim=1)
            
            # Stop if we generated an EOS token
            if next_token == self.small_tokenizer.eos_token_id:
                break
        
        return draft_ids, draft_probs
    
    def _verify_draft_tokens(self, context_ids: torch.Tensor, draft_ids: List[int], 
                           draft_probs: List[float]) -> Tuple[torch.Tensor, List[float], List[bool]]:
        """
        Verify draft tokens using the large model.
        
        Args:
            context_ids: Current context tokens
            draft_ids: List of draft token IDs generated by small model
            draft_probs: List of probabilities for draft tokens
            
        Returns:
            Tuple of (accepted token IDs, large model probabilities, acceptance mask)
        """
        if not draft_ids:
            return torch.tensor([]), [], []
        
        # Prepare draft tokens for verification
        draft_tokens = torch.tensor(draft_ids).unsqueeze(0).to(context_ids.device)
        
        # Concatenate context with draft tokens
        full_seq = torch.cat([context_ids, draft_tokens], dim=1)
        
        # Get logits from the large model
        with torch.no_grad():
            outputs = self.large_model(full_seq)
            
        # Extract logits for the positions corresponding to draft tokens
        logits = outputs.logits[:, context_ids.shape[1]-1:-1, :]
        probs = torch.softmax(logits, dim=-1)
        
        # Get probabilities assigned by the large model to each draft token
        large_model_probs = []
        for i, token_id in enumerate(draft_ids):
            large_model_probs.append(probs[0, i, token_id].item())
        
        # Accept or reject tokens based on the verification
        accepted_mask = []
        for q_j, p_j in zip(large_model_probs, draft_probs):
            accepted = q_j >= self.config.gamma * p_j
            accepted_mask.append(accepted)
            
            # Stop at the first rejection
            if not accepted:
                break
        
        # Keep only the accepted tokens
        num_accepted = sum(accepted_mask)
        accepted_ids = torch.tensor(draft_ids[:num_accepted], device=context_ids.device)
        
        return accepted_ids, large_model_probs, accepted_mask
    
    def _generate_single_token(self, input_ids: torch.Tensor) -> Tuple[Optional[int], float]:
        """
        Generate a single token using the large model.
        
        Args:
            input_ids: Current context tokens
            
        Returns:
            Tuple of (token ID, probability)
        """
        with torch.no_grad():
            outputs = self.large_model(input_ids)

        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        
        # Sample a token
        next_token = torch.multinomial(probs[0], 1).item()
        next_token_prob = probs[0, next_token].item()
        
        return next_token, next_token_prob
    
    def _record_token_info(self, draft_ids: List[int], accepted_ids: torch.Tensor, 
                         accepted_mask: List[bool], draft_probs: List[float], 
                         large_model_probs: List[float]):
        """
        Record token information for analysis.
        
        Args:
            draft_ids: List of draft token IDs
            accepted_ids: Tensor of accepted token IDs
            accepted_mask: List of boolean flags indicating acceptance
            draft_probs: List of probabilities from small model
            large_model_probs: List of probabilities from large model
        """
        for i, token_id in enumerate(draft_ids):
            # Get the decoded token text
            token_text = self.small_tokenizer.decode([token_id])
            
            # Record accepted tokens
            if i < len(accepted_mask) and accepted_mask[i]:
                self._record_token(
                    token_text=token_text,
                    source="small_model",
                    prob_small=draft_probs[i],
                    prob_big=large_model_probs[i],
                    rejection_reason=None
                )
            # Record rejected tokens
            elif i < len(accepted_mask):
                rejection_reason = "probability_threshold"
                if large_model_probs[i] < self.config.gamma * draft_probs[i]:
                    rejection_reason = "probability_threshold"
                
                self._record_token(
                    token_text=token_text,
                    source="small_model",
                    prob_small=draft_probs[i],
                    prob_big=large_model_probs[i],
                    rejection_reason=rejection_reason
                )


# ==========================================================
# Analysis and Visualization
# ==========================================================

class ResultsAnalyzer:
    """Analyzes and visualizes the results of speculative decoding."""
    
    def __init__(self, config: Config):
        """Initialize the analyzer with the configuration."""
        self.config = config
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.visualization_dir, exist_ok=True)
        
        # For aggregate analysis across problems
        self.aggregate_data = []
    
    def save_to_csv(self, problem_id: str, token_records: List[Dict[str, Any]]):
        """
        Save token records to a CSV file.
        
        Args:
            problem_id: ID of the problem
            token_records: List of token records
        """
        # Create a problem-specific directory
        problem_dir = os.path.join(self.config.output_dir, str(problem_id))
        os.makedirs(problem_dir, exist_ok=True)
        
        # Save to CSV
        csv_path = os.path.join(problem_dir, self.config.csv_filename)
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                "token_id", "token_text", "source", "probability_small", 
                "probability_big", "rejection_reason"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for record in token_records:
                writer.writerow(record)
        
        print(f"Saved token analysis to {csv_path}")
        
        # Add to aggregate data
        for record in token_records:
            self.aggregate_data.append({
                **record,
                "problem_id": problem_id
            })
        
        return csv_path
    
    def visualize_results(self, problem_id: str, csv_path: str):
        """
        Create visualizations from token records.
        
        Args:
            problem_id: ID of the problem
            csv_path: Path to the CSV file with token records
        """
        # Load data
        df = pd.read_csv(csv_path)
        
        # Create visualizations directory for this problem
        vis_dir = os.path.join(self.config.visualization_dir, str(problem_id))
        os.makedirs(vis_dir, exist_ok=True)
        
        # Add token type classification
        self._classify_token_types(df)
        
        # 1. Token source distribution
        self._plot_token_source_distribution(df, vis_dir)
        
        # 2. Acceptance rate over generation
        self._plot_acceptance_rate(df, vis_dir)
        
        # 3. Probability comparison between models
        self._plot_probability_comparison(df, vis_dir)
        
        # 4. Rejection reasons
        self._plot_rejection_reasons(df, vis_dir)
        
        # 5. Token type analysis
        self._plot_token_type_analysis(df, vis_dir)
        
        # 6. Math symbol analysis
        self._plot_math_symbol_analysis(df, vis_dir)
        
        # 7. Probability histogram
        self._plot_probability_histograms(df, vis_dir)
        
        # 8. Token source by token ID (New plot)
        self._plot_token_source_by_id(df, vis_dir)
    
    def save_aggregate_results(self):
        """
        Save and visualize aggregate results across all problems.
        """
        if not self.aggregate_data:
            print("No aggregate data available")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.aggregate_data)
        
        # Create aggregate directory
        agg_dir = os.path.join(self.config.visualization_dir, "aggregate")
        os.makedirs(agg_dir, exist_ok=True)
        
        # Save to CSV
        aggregate_csv = os.path.join(self.config.output_dir, "aggregate_results.csv")
        df.to_csv(aggregate_csv, index=False)
        print(f"Saved aggregate results to {aggregate_csv}")
        
        # Add token type classification
        self._classify_token_types(df)
        
        # Generate aggregate visualizations
        self._plot_aggregate_statistics(df, agg_dir)
        
        return aggregate_csv
    
    def _classify_token_types(self, df: pd.DataFrame):
        """
        Classify tokens into different types (math symbols, numbers, text, etc.).
        """
        # Math symbols regex patterns
        math_symbols = r'[+\-*/=<>≤≥≠∈∉⊂⊃∩∪∀∃∄∑∏∫∂∇√∞±≈≡≢|]'
        numbers = r'\d+'
        variables = r'\b[a-zA-Z]\b'  # Single letters likely to be variables
        
        # Classify tokens
        df['is_math_symbol'] = df['token_text'].str.contains(math_symbols, regex=True)
        df['is_number'] = df['token_text'].str.contains(numbers, regex=True)
        df['is_variable'] = df['token_text'].str.contains(variables, regex=True)
        
        # Create token type categories
        conditions = [
            df['is_math_symbol'],
            df['is_number'],
            df['is_variable'],
            df['token_text'].str.strip() == ''
        ]
        choices = ['math_symbol', 'number', 'variable', 'whitespace']
        df['token_type'] = np.select(conditions, choices, default='text')
    
    def _plot_token_source_distribution(self, df: pd.DataFrame, output_dir: str):
        """Plot the distribution of token sources (small vs large model)."""
        plt.figure(figsize=(10, 6))
        counts = df['source'].value_counts()
        ax = counts.plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
        plt.title('Token Source Distribution')
        plt.xlabel('Model Source')
        plt.ylabel('Number of Tokens')
        plt.xticks(rotation=0)
        
        # Add counts on top of bars
        for i, count in enumerate(counts):
            ax.text(i, count + 5, str(count), ha='center')
        
        output_path = os.path.join(output_dir, 'token_source_distribution.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _plot_acceptance_rate(self, df: pd.DataFrame, output_dir: str):
        """Plot the acceptance rate over the course of generation."""
        plt.figure(figsize=(12, 6))
        
        # Create a column for accepted/rejected
        df['accepted'] = df['rejection_reason'].isna() & (df['source'] == 'small_model')
        
        # Group by chunks of tokens to see acceptance rate over time
        chunk_size = 20
        df['token_chunk'] = df['token_id'] // chunk_size
        
        # Calculate acceptance rate per chunk
        acceptance_rates = df[df['source'] == 'small_model'].groupby('token_chunk')['accepted'].mean()
        
        plt.plot(acceptance_rates.index, acceptance_rates.values, marker='o', linestyle='-')
        plt.title('Small Model Token Acceptance Rate During Generation')
        plt.xlabel(f'Generation Chunk (each chunk = {chunk_size} tokens)')
        plt.ylabel('Acceptance Rate')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(output_dir, 'acceptance_rate.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _plot_probability_comparison(self, df: pd.DataFrame, output_dir: str):
        """Plot comparison between small and large model probabilities."""
        # Filter to only include tokens where we have both probabilities
        mask = df['probability_big'].notna()
        df_both = df[mask].copy()
        
        plt.figure(figsize=(10, 8))
        plt.scatter(
            df_both['probability_small'], 
            df_both['probability_big'],
            c=df_both['rejection_reason'].isna(),  # Color by acceptance
            cmap='coolwarm',
            alpha=0.7
        )
        
        # Add diagonal line (perfect alignment)
        max_val = max(df_both['probability_small'].max(), df_both['probability_big'].max())
        min_val = min(df_both['probability_small'].min(), df_both['probability_big'].min())
        plt.plot([min_val, max_val], [min_val, max_val], 'g--', alpha=0.7)
        
        # Add threshold line
        x = np.linspace(min_val, max_val, 100)
        y = self.config.gamma * x
        plt.plot(x, y, 'r--', alpha=0.7, label=f'Acceptance Threshold (γ={self.config.gamma})')
        
        plt.title('Small vs Large Model Token Probabilities')
        plt.xlabel('Small Model Probability')
        plt.ylabel('Large Model Probability')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        output_path = os.path.join(output_dir, 'probability_comparison.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _plot_rejection_reasons(self, df: pd.DataFrame, output_dir: str):
        """Plot distribution of rejection reasons."""
        # Add "accepted" as a reason
        df_plot = df.copy()
        df_plot.loc[df_plot['rejection_reason'].isna() & (df_plot['source'] == 'small_model'), 'rejection_reason'] = 'accepted'
        df_plot.loc[df_plot['source'] == 'big_model', 'rejection_reason'] = 'big_model_generated'
        
        plt.figure(figsize=(10, 6))
        counts = df_plot['rejection_reason'].value_counts()
        ax = counts.plot(kind='bar', color=['#2ca02c', '#d62728', '#9467bd'])
        plt.title('Token Acceptance and Rejection Distribution')
        plt.xlabel('Status')
        plt.ylabel('Number of Tokens')
        plt.xticks(rotation=45)
        
        # Add counts on top of bars
        for i, count in enumerate(counts):
            ax.text(i, count + 5, str(count), ha='center')
        
        output_path = os.path.join(output_dir, 'rejection_reasons.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _plot_token_type_analysis(self, df: pd.DataFrame, output_dir: str):
        """Analyze and visualize token types."""
        plt.figure(figsize=(12, 8))
        
        # Plot acceptance rate by token type for small model tokens
        small_model_df = df[df['source'] == 'small_model'].copy()
        small_model_df['accepted'] = small_model_df['rejection_reason'].isna()
        
        token_type_acceptance = small_model_df.groupby('token_type')['accepted'].agg(['mean', 'count']).reset_index()
        token_type_acceptance.columns = ['token_type', 'acceptance_rate', 'count']
        
        # Sort by count for better visualization
        token_type_acceptance = token_type_acceptance.sort_values('count', ascending=False)
        
        ax = token_type_acceptance.plot(
            x='token_type',
            y='acceptance_rate',
            kind='bar',
            color='skyblue',
            figsize=(12, 6)
        )
        
        # Add count annotations
        for i, row in enumerate(token_type_acceptance.itertuples()):
            ax.text(i, row.acceptance_rate + 0.02, f'n={row.count}', ha='center')
        
        plt.title('Token Acceptance Rate by Token Type')
        plt.xlabel('Token Type')
        plt.ylabel('Acceptance Rate')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        output_path = os.path.join(output_dir, 'token_type_acceptance.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        # Distribution of token types
        plt.figure(figsize=(10, 6))
        token_counts = df['token_type'].value_counts()
        token_counts.plot(kind='pie', autopct='%1.1f%%', colors=plt.cm.tab10.colors)
        plt.title('Distribution of Token Types')
        plt.ylabel('')
        
        output_path = os.path.join(output_dir, 'token_type_distribution.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _plot_math_symbol_analysis(self, df: pd.DataFrame, output_dir: str):
        """Analyze and visualize math symbol usage."""
        # Only analyze if there are math symbols
        if not any(df['is_math_symbol']):
            return
        
        # Analyze rejection rate for math symbols vs non-math symbols
        small_model_df = df[df['source'] == 'small_model'].copy()
        small_model_df['accepted'] = small_model_df['rejection_reason'].isna()
        
        # Group by math symbol flag
        math_acceptance = small_model_df.groupby('is_math_symbol')['accepted'].agg(['mean', 'count']).reset_index()
        math_acceptance.columns = ['is_math_symbol', 'acceptance_rate', 'count']
        
        plt.figure(figsize=(10, 6))
        ax = plt.bar(
            [f"Math Symbol (n={math_acceptance.loc[1, 'count']})" if x else f"Non-Math Symbol (n={math_acceptance.loc[0, 'count']})" 
             for x in math_acceptance['is_math_symbol']],
            math_acceptance['acceptance_rate'],
            color=['#ff9999', '#66b3ff']
        )
        
        plt.title('Acceptance Rate: Math Symbols vs Other Tokens')
        plt.xlabel('Token Type')
        plt.ylabel('Acceptance Rate')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        # Add percentage annotations
        for i, p in enumerate(ax):
            height = p.get_height()
            plt.text(p.get_x() + p.get_width()/2.,
                    height + 0.02,
                    f'{height:.1%}',
                    ha='center')
        
        output_path = os.path.join(output_dir, 'math_symbol_acceptance.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _plot_probability_histograms(self, df: pd.DataFrame, output_dir: str):
        """Plot histograms of token probabilities."""
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Filter for accepted and rejected tokens from small model
        accepted = df[(df['source'] == 'small_model') & (df['rejection_reason'].isna())]
        rejected = df[(df['source'] == 'small_model') & (df['rejection_reason'].notna())]
        
        # Plot histogram of small model probabilities
        ax1.hist(accepted['probability_small'], bins=20, alpha=0.7, label='Accepted', color='green')
        ax1.hist(rejected['probability_small'], bins=20, alpha=0.7, label='Rejected', color='red')
        ax1.set_title('Small Model Probability Distribution')
        ax1.set_xlabel('Probability')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot histogram of large model probabilities
        if not rejected.empty and 'probability_big' in rejected.columns:
            ax2.hist(accepted['probability_big'], bins=20, alpha=0.7, label='Accepted', color='green')
            ax2.hist(rejected['probability_big'], bins=20, alpha=0.7, label='Rejected', color='red')
            ax2.set_title('Large Model Probability Distribution')
            ax2.set_xlabel('Probability')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'probability_histograms.png')
        plt.savefig(output_path)
        plt.close()
    
    def _plot_aggregate_statistics(self, df: pd.DataFrame, output_dir: str):
        """
        Generate aggregate statistics and visualizations across all problems.
        
        Args:
            df: DataFrame with all token records from all problems
            output_dir: Directory to save visualizations
        """
        # Classify token types if not already done
        if 'token_type' not in df.columns:
            self._classify_token_types(df)
        
        # 1. Overall acceptance rate across problems
        acceptance_by_problem = df[df['source'] == 'small_model'].groupby('problem_id')['rejection_reason'].apply(
            lambda x: x.isna().mean()
        ).reset_index()
        acceptance_by_problem.columns = ['problem_id', 'acceptance_rate']
        
        plt.figure(figsize=(12, 6))
        ax = acceptance_by_problem.plot(
            kind='bar', 
            x='problem_id', 
            y='acceptance_rate',
            color='skyblue'
        )
        plt.title('Token Acceptance Rate by Problem')
        plt.xlabel('Problem ID')
        plt.ylabel('Acceptance Rate')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add rate annotations
        for i, row in enumerate(acceptance_by_problem.itertuples()):
            ax.text(i, row.acceptance_rate + 0.02, f'{row.acceptance_rate:.2f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'acceptance_by_problem.png'))
        plt.close()
        
        # 2. Token type distribution across problems
        token_type_counts = df.groupby(['problem_id', 'token_type']).size().unstack(fill_value=0)
        token_type_counts.plot(
            kind='bar',
            stacked=True,
            figsize=(14, 8),
            colormap='tab10'
        )
        plt.title('Token Type Distribution by Problem')
        plt.xlabel('Problem ID')
        plt.ylabel('Number of Tokens')
        plt.legend(title='Token Type')
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'token_types_by_problem.png'))
        plt.close()
        
        # 3. Compute and save speedup by problem
        df['is_accepted'] = df['rejection_reason'].isna() & (df['source'] == 'small_model')
        df['is_rejected'] = df['rejection_reason'].notna()
        df['is_big_model'] = df['source'] == 'big_model'
        
        speedup_by_problem = df.groupby('problem_id').apply(
            lambda x: len(x) / (sum(x['is_rejected']) + sum(x['is_big_model']))
        ).reset_index()
        speedup_by_problem.columns = ['problem_id', 'theoretical_speedup']
        
        plt.figure(figsize=(12, 6))
        ax = speedup_by_problem.plot(
            kind='bar',
            x='problem_id',
            y='theoretical_speedup',
            color='orange'
        )
        plt.title('Theoretical Speedup by Problem')
        plt.xlabel('Problem ID')
        plt.ylabel('Speedup Factor')
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add speedup annotations
        for i, row in enumerate(speedup_by_problem.itertuples()):
            ax.text(i, row.theoretical_speedup + 0.1, f'{row.theoretical_speedup:.2f}x', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'speedup_by_problem.png'))
        plt.close()
        
        # Save the summary statistics to CSV
        summary_stats = pd.DataFrame({
            'problem_id': speedup_by_problem['problem_id'],
            'acceptance_rate': acceptance_by_problem['acceptance_rate'],
            'theoretical_speedup': speedup_by_problem['theoretical_speedup']
        })
        
        summary_stats.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)
        
        return summary_stats

    def _plot_token_source_by_id(self, df: pd.DataFrame, output_dir: str):
        """
        Plot token source (big or small model) by token ID to visualize the pattern of token generation.
        """
        plt.figure(figsize=(14, 6))
        
        # Create a categorical y-axis for the model source
        # Map 'small_model' to 0 and 'big_model' to 1 for plotting
        source_map = {'small_model': 0, 'big_model': 1}
        df['source_numeric'] = df['source'].map(source_map)
        
        # Add color based on whether the small model token was accepted or rejected
        colors = []
        for _, row in df.iterrows():
            if row['source'] == 'small_model':
                # Green for accepted tokens from small model, red for rejected
                colors.append('green' if pd.isna(row['rejection_reason']) else 'red')
            else:
                # Blue for tokens from big model
                colors.append('blue')
        
        # Create the scatter plot
        plt.scatter(df['token_id'], df['source_numeric'], c=colors, s=50, alpha=0.7)
        
        # Customize the plot
        plt.yticks([0, 1], ['Small Model', 'Big Model'])
        plt.title('Token Source by Token ID')
        plt.xlabel('Token ID (Generation Sequence)')
        plt.ylabel('Source Model')
        plt.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Small Model (Accepted)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Small Model (Rejected)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Big Model')
        ]
        plt.legend(handles=legend_elements, loc='best')
        
        # Save the figure
        output_path = os.path.join(output_dir, 'token_source_by_id.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


# ==========================================================
# Main Execution
# ==========================================================

def main():
    """Main execution function."""
    # Load configuration
    config = Config()
    
    # Load dataset
    print("Loading AIME problems dataset...")
    dataset = AIMEDataset(config)
    print(f"Loaded {len(dataset)} problems")
    
    # Initialize speculative decoder
    print("Initializing speculative decoder...")
    decoder = SpeculativeDecoder(config)
    
    # Initialize results analyzer
    analyzer = ResultsAnalyzer(config)
    
    # Process each problem
    for idx in range(len(dataset)):
        problem = dataset[idx]
        problem_id = problem["id"]
        problem_text = problem["text"]
        
        print(f"\nProcessing problem {problem_id}...")
        print(f"Problem: {problem_text}")
        
        # Construct prompt using the template from config
        prompt = config.prompt_template.format(
            problem=problem_text,
            terminating_string=config.terminating_string,
            model_think_prefix=config.model_think_prefix
        )
        
        # Perform speculative decoding
        print("Generating solution with speculative decoding...")
        start_time = time.time()
        solution, token_records = decoder.speculative_decode(prompt)
        end_time = time.time()
        
        # Print statistics
        total_tokens = len(token_records)
        small_tokens = sum(1 for record in token_records if record["source"] == "small_model" and record["rejection_reason"] is None)
        large_tokens = sum(1 for record in token_records if record["source"] == "big_model")
        rejected_tokens = sum(1 for record in token_records if record["rejection_reason"] is not None)
        
        print(f"Generation complete in {end_time - start_time:.2f} seconds")
        print(f"Total tokens: {total_tokens}")
        print(f"Tokens from small model (accepted): {small_tokens} ({small_tokens/total_tokens*100:.2f}%)")
        print(f"Tokens from large model: {large_tokens} ({large_tokens/total_tokens*100:.2f}%)")
        print(f"Rejected tokens: {rejected_tokens} ({rejected_tokens/(small_tokens+rejected_tokens)*100:.2f}% of small model tokens)")
        
        # Calculate speedup
        theoretical_speedup = total_tokens / (rejected_tokens + large_tokens) if (rejected_tokens + large_tokens) > 0 else float('inf')
        print(f"Theoretical speedup: {theoretical_speedup:.2f}x")
        
        # Save results
        print(f"Saving results for problem {problem_id}...")
        csv_path = analyzer.save_to_csv(problem_id, token_records)
        
        # Create visualizations
        print("Creating visualizations...")
        analyzer.visualize_results(problem_id, csv_path)
        
        # Print the generated solution
        print("\nGenerated solution:")
        print(solution)
        print("\n" + "="*80)
    
    # Generate and save aggregate statistics
    print("\nGenerating aggregate statistics...")
    analyzer.save_aggregate_results()
    
    print("\nSpeculative decoding experiment complete!")


if __name__ == "__main__":
    import numpy as np  # Needed for analyzer
    main()


if __name__ == "__main__":
    main()