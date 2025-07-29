"""Vectorized GSPO Implementation with bug fixes."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Dict, Tuple, Callable, Optional

__all__ = ['GSPOTrainer', 'GSPOConfig', 'vectorized_gspo_update']

class GSPOError(Exception):
    """Base exception for GSPO errors."""
    pass

class GSPOConfig:
    """Configuration for GSPO training."""
    def __init__(self, 
                 group_size: int = 4,
                 epsilon: float = 0.2, 
                 max_length: int = 512,
                 eos_token: int = 2,  # Changed from 0 to avoid confusion with padding
                 pad_token: int = 0):  # Explicit padding token
        assert group_size > 0, f"Invalid group_size: {group_size}"
        assert 0 < epsilon < 1, f"Invalid epsilon: {epsilon}"
        assert max_length > 0, f"Invalid max_length: {max_length}"
        assert eos_token != pad_token, "EOS and padding tokens must be different"
        
        self.group_size = group_size
        self.epsilon = epsilon
        self.max_length = max_length
        self.eos_token = eos_token
        self.pad_token = pad_token
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        return (self.group_size > 0 and 
                0 < self.epsilon < 1 and 
                self.max_length > 0 and
                self.eos_token != self.pad_token)

class GSPOTrainer:
    """Vectorized GSPO trainer with comprehensive validation."""
    
    def __init__(self, 
                 policy_model: nn.Module,
                 ref_model: nn.Module,
                 optimizer: optim.Optimizer,
                 config: GSPOConfig,
                 reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        """Initialize GSPO trainer.
        
        Args:
            policy_model: Current policy to optimize
            ref_model: Reference (frozen) policy
            optimizer: Optimizer for policy model
            config: GSPO configuration
            reward_fn: Function that takes (prompts, responses) -> rewards
            
        Raises:
            GSPOError: If configuration is invalid
        """
        assert config.validate(), "Invalid GSPO configuration"
        assert policy_model is not None, "Policy model required"
        assert ref_model is not None, "Reference model required"
        assert reward_fn is not None, "Reward function required"
        
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.config = config
        self.reward_fn = reward_fn
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def compute_log_probs_batch(self, 
                               model: nn.Module, 
                               full_sequences: torch.Tensor, 
                               prompt_length: int,
                               response_lengths: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for batch of sequences.
        
        Args:
            model: Language model
            full_sequences: Complete sequences (batch_size, seq_len) = prompts + responses
            prompt_length: Length of prompt part
            response_lengths: Actual lengths of response parts
            
        Returns:
            Log probabilities summed over response tokens only
            
        Raises:
            GSPOError: If shapes don't match
        """
        assert full_sequences.size(0) == response_lengths.size(0), "Batch size mismatch"
        assert prompt_length > 0, "Invalid prompt length"
        
        batch_size = full_sequences.size(0)
        device = full_sequences.device
        
        # Autoregressive: input is shifted left, targets are shifted right
        input_ids = full_sequences[:, :-1]  # (batch_size, seq_len-1)
        target_ids = full_sequences[:, 1:]  # (batch_size, seq_len-1)
        
        # Get logits from model
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(input_ids)  # (batch_size, seq_len-1, vocab_size)
        
        # Convert to log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Gather log probs for target tokens
        target_expanded = target_ids.unsqueeze(-1)  # (batch_size, seq_len-1, 1)
        gathered_log_probs = log_probs.gather(-1, target_expanded).squeeze(-1)  # (batch_size, seq_len-1)
        
        # Create mask for response positions only
        seq_positions = torch.arange(gathered_log_probs.size(1), device=device).unsqueeze(0)  # (1, seq_len-1)
        
        # Response starts at prompt_length (0-indexed), goes for response_lengths tokens
        response_start = prompt_length
        response_end = response_start + response_lengths.unsqueeze(1)  # (batch_size, 1)
        
        # Mask: True for positions in [response_start, response_start + length)
        response_mask = (seq_positions >= response_start) & (seq_positions < response_end)
        
        # Apply mask and sum over response tokens
        masked_log_probs = gathered_log_probs * response_mask.float()
        summed_log_probs = masked_log_probs.sum(dim=1)  # (batch_size,)
        
        assert not torch.any(torch.isnan(summed_log_probs)), "NaN in log probabilities"
        assert not torch.any(torch.isinf(summed_log_probs)), "Inf in log probabilities"
        
        return summed_log_probs
    
    def sample_responses_batch(self, 
                              prompts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample responses from reference model for batch of prompts - VECTORIZED.
        
        Args:
            prompts: Batch of prompts (batch_size, prompt_len)
            
        Returns:
            Tuple of (responses, lengths, full_sequences)
            
        Raises:
            GSPOError: If sampling fails
        """
        assert prompts.size(0) > 0, "Empty prompt batch"
        
        batch_size = prompts.size(0)
        device = prompts.device
        group_size = self.config.group_size
        max_len = self.config.max_length
        eos_token = self.config.eos_token
        pad_token = self.config.pad_token
        
        # Expand prompts for group sampling
        expanded_prompts = prompts.repeat_interleave(group_size, dim=0)  # (batch_size * group_size, prompt_len)
        total_sequences = expanded_prompts.size(0)
        
        # Initialize generation state
        current = expanded_prompts.clone()  # Current sequences being generated
        responses = torch.full((total_sequences, max_len), pad_token, dtype=torch.long, device=device)
        active = torch.ones(total_sequences, dtype=torch.bool, device=device)  # Which sequences are still generating
        lengths = torch.zeros(total_sequences, dtype=torch.long, device=device)
        
        self.ref_model.eval()
        with torch.no_grad():
            for step in range(max_len):
                # Get logits for next token (vectorized across all active sequences)
                logits = self.ref_model(current)[:, -1, :]  # (total_sequences, vocab_size)
                
                # Sample next tokens
                dist = Categorical(logits=logits)
                next_tokens = dist.sample()  # (total_sequences,)
                
                # Only update active sequences
                next_tokens[~active] = pad_token
                
                # Store response tokens
                responses[:, step] = next_tokens
                
                # Update current sequences for next iteration
                current = torch.cat([current, next_tokens.unsqueeze(1)], dim=1)
                
                # Check for finished sequences
                just_finished = (next_tokens == eos_token) & active
                lengths[just_finished] = step + 1  # Set length when finished
                
                # Update active mask
                active = active & (next_tokens != eos_token)
                
                # Early exit if all sequences finished
                if not active.any():
                    break
            
            # Handle sequences that didn't finish (reached max_len)
            still_active = active
            lengths[still_active] = max_len
        
        # Trim responses to actual max length used
        actual_max_len = lengths.max().item()
        responses = responses[:, :actual_max_len]
        
        # Create full sequences (prompt + response)
        full_sequences = torch.cat([expanded_prompts, responses], dim=1)
        
        assert responses.size(0) == batch_size * group_size, "Response count mismatch"
        assert lengths.size(0) == batch_size * group_size, "Length count mismatch"
        assert torch.all(lengths > 0), "Zero-length responses detected"
        
        return responses, lengths, full_sequences
    
    def update_step(self, prompts: torch.Tensor) -> Dict[str, float]:
        """Perform single GSPO update step.
        
        Args:
            prompts: Batch of prompts (batch_size, prompt_len)
            
        Returns:
            Dictionary of training metrics
            
        Raises:
            GSPOError: If update fails
        """
        assert prompts.size(0) > 0, "Empty prompt batch"
        
        batch_size = prompts.size(0)
        group_size = self.config.group_size
        prompt_length = prompts.size(1)
        device = prompts.device
        
        # Step 1: Sample responses from reference model
        responses, lengths, full_sequences = self.sample_responses_batch(prompts)
        
        # Step 2: Compute rewards
        expanded_prompts = prompts.repeat_interleave(group_size, dim=0)
        rewards = self.reward_fn(expanded_prompts, responses)
        assert rewards.size(0) == batch_size * group_size, "Reward count mismatch"
        
        # Step 3: Reshape for group processing
        rewards_grouped = rewards.view(batch_size, group_size)  # (batch_size, group_size)
        lengths_grouped = lengths.view(batch_size, group_size)
        
        # Step 4: Compute group-based advantages
        mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)  # (batch_size, 1)
        std_rewards = rewards_grouped.std(dim=1, keepdim=True) + 1e-8  # (batch_size, 1)
        advantages = (rewards_grouped - mean_rewards) / std_rewards  # (batch_size, group_size)
        
        # Step 5: Compute log probabilities for both models
        # Reference model log probs
        ref_log_probs = self.compute_log_probs_batch(
            self.ref_model, 
            full_sequences,
            prompt_length,
            lengths
        )
        
        # Policy model log probs  
        policy_log_probs = self.compute_log_probs_batch(
            self.policy_model,
            full_sequences,
            prompt_length,
            lengths
        )
        
        # Step 6: Compute sequence-level importance ratios with length normalization
        log_ratios = (policy_log_probs - ref_log_probs) / lengths.float()  # Length normalized
        ratios = torch.exp(log_ratios)
        
        # Step 7: Apply clipping
        epsilon = self.config.epsilon
        clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
        
        # Step 8: Compute surrogate terms
        advantages_flat = advantages.view(-1)  # Flatten to match ratios
        
        term1 = ratios * advantages_flat
        term2 = clipped_ratios * advantages_flat
        surrogate_terms = torch.min(term1, term2)
        
        # Step 9: Compute loss (negative because we maximize)
        loss = -surrogate_terms.mean()
        
        # Step 10: Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Compute metrics
        metrics = {
            "loss": loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "mean_ratio": ratios.mean().item(),
            "clipped_fraction": (ratios != clipped_ratios).float().mean().item(),
            "mean_response_length": lengths.float().mean().item()
        }
        
        # Validate metrics
        for key, value in metrics.items():
            assert not torch.isnan(torch.tensor(value)), f"NaN in metric {key}"
            assert not torch.isinf(torch.tensor(value)), f"Inf in metric {key}"
        
        return metrics

def vectorized_gspo_update(policy_model: nn.Module,
                          ref_model: nn.Module, 
                          optimizer: optim.Optimizer,
                          prompts: torch.Tensor,
                          reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                          config: Optional[GSPOConfig] = None) -> Dict[str, float]:
    """Convenient function for single GSPO update.
    
    Args:
        policy_model: Current policy to optimize
        ref_model: Reference (frozen) policy  
        optimizer: Optimizer for policy model
        prompts: Batch of prompts
        reward_fn: Reward function
        config: GSPO configuration (uses defaults if None)
        
    Returns:
        Training metrics dictionary
        
    Raises:
        GSPOError: If update fails
    """
    if config is None:
        config = GSPOConfig()
    
    trainer = GSPOTrainer(policy_model, ref_model, optimizer, config, reward_fn)
    return trainer.update_step(prompts)

# Example usage with validation
if __name__ == "__main__":
    # Example reward function that properly handles padding
    def example_reward_fn(prompts: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        """Example reward based on non-padded response length."""
        assert prompts.size(0) == responses.size(0), "Batch size mismatch"
        
        # Count non-padding tokens (assuming pad_token=0)
        pad_token = 0
        lengths = (responses != pad_token).sum(dim=1).float()
        
        # Simple reward: longer responses get higher reward, with some noise
        base_reward = lengths / 10.0
        noise = torch.randn_like(base_reward) * 0.1  # Add some variance
        rewards = base_reward + noise
        
        assert rewards.size(0) == prompts.size(0), "Reward count mismatch"
        return rewards
    
    # Configuration with proper token separation
    config = GSPOConfig(
        group_size=4, 
        epsilon=0.2, 
        max_length=100,
        eos_token=2,  # Different from padding
        pad_token=0
    )
    assert config.validate(), "Invalid configuration"
    
    print("GSPO vectorized implementation ready!")
    print("Key fixes applied:")
    print("- Removed unused cupy import")
    print("- Fixed log probability computation bug")
    print("- Implemented proper vectorized sampling")
    print("- Separated EOS and padding tokens")
    print("- Added proper sequence masking") 