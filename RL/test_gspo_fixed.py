"""Test script for the fixed GSPO implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
from gspo_vectorized import GSPOTrainer, GSPOConfig, vectorized_gspo_update

class DummyLanguageModel(nn.Module):
    """Simple dummy language model for testing."""
    
    def __init__(self, vocab_size: int = 1000, hidden_size: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )
        self.output = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        # Simple causal mask for autoregressive generation
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        if input_ids.is_cuda:
            causal_mask = causal_mask.cuda()
            
        x = self.embedding(input_ids)
        x = self.transformer(x, x, tgt_mask=causal_mask)
        logits = self.output(x)
        return logits

def test_reward_fn(prompts: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
    """Test reward function: reward based on length and some randomness."""
    # Count non-padding tokens (pad_token = 0)
    lengths = (responses != 0).sum(dim=1).float()
    
    # Base reward from length
    base_reward = lengths / 10.0
    
    # Add some random variation to make it interesting
    noise = torch.randn_like(base_reward) * 0.1
    rewards = base_reward + noise
    
    return rewards

def run_gspo_test():
    """Run comprehensive GSPO test."""
    print("🧪 Testing Fixed GSPO Implementation")
    print("=" * 50)
    
    # Test configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    vocab_size = 100  # Small vocab for testing
    batch_size = 2
    prompt_length = 5
    
    # Create models
    print("\n📦 Creating models...")
    policy_model = DummyLanguageModel(vocab_size=vocab_size, hidden_size=32).to(device)
    ref_model = DummyLanguageModel(vocab_size=vocab_size, hidden_size=32).to(device)
    
    # Copy weights to make ref model identical initially
    ref_model.load_state_dict(policy_model.state_dict())
    
    # Create optimizer
    optimizer = optim.Adam(policy_model.parameters(), lr=1e-4)
    
    # GSPO configuration
    config = GSPOConfig(
        group_size=2,  # Small for testing
        epsilon=0.2,
        max_length=8,  # Short for testing
        eos_token=1,   # Different from padding
        pad_token=0
    )
    
    print(f"Config: group_size={config.group_size}, epsilon={config.epsilon}, max_length={config.max_length}")
    
    # Create test prompts (random tokens, avoiding special tokens)
    prompts = torch.randint(2, vocab_size, (batch_size, prompt_length), device=device)
    print(f"\n📝 Test prompts shape: {prompts.shape}")
    print(f"Sample prompt: {prompts[0].tolist()}")
    
    # Test 1: Validate configuration
    print("\n✅ Test 1: Configuration validation")
    assert config.validate(), "Configuration should be valid"
    print("Configuration validation passed!")
    
    # Test 2: Test trainer initialization
    print("\n✅ Test 2: Trainer initialization")
    trainer = GSPOTrainer(policy_model, ref_model, optimizer, config, test_reward_fn)
    print("Trainer initialization passed!")
    
    # Test 3: Test response sampling
    print("\n✅ Test 3: Response sampling")
    responses, lengths, full_sequences = trainer.sample_responses_batch(prompts)
    
    expected_total = batch_size * config.group_size
    print(f"Generated responses shape: {responses.shape}")
    print(f"Lengths shape: {lengths.shape}")
    print(f"Full sequences shape: {full_sequences.shape}")
    print(f"Sample response: {responses[0].tolist()}")
    print(f"Sample length: {lengths[0].item()}")
    
    assert responses.size(0) == expected_total, f"Expected {expected_total} responses, got {responses.size(0)}"
    assert lengths.size(0) == expected_total, f"Expected {expected_total} lengths, got {lengths.size(0)}"
    assert torch.all(lengths > 0), "All responses should have positive length"
    print("Response sampling passed!")
    
    # Test 4: Test log probability computation
    print("\n✅ Test 4: Log probability computation")
    
    ref_log_probs = trainer.compute_log_probs_batch(
        ref_model, full_sequences, prompt_length, lengths
    )
    policy_log_probs = trainer.compute_log_probs_batch(
        policy_model, full_sequences, prompt_length, lengths
    )
    
    print(f"Ref log probs shape: {ref_log_probs.shape}")
    print(f"Policy log probs shape: {policy_log_probs.shape}")
    print(f"Sample ref log prob: {ref_log_probs[0].item():.4f}")
    print(f"Sample policy log prob: {policy_log_probs[0].item():.4f}")
    
    assert ref_log_probs.size(0) == expected_total, "Ref log probs size mismatch"
    assert policy_log_probs.size(0) == expected_total, "Policy log probs size mismatch"
    assert not torch.any(torch.isnan(ref_log_probs)), "NaN in ref log probs"
    assert not torch.any(torch.isnan(policy_log_probs)), "NaN in policy log probs"
    print("Log probability computation passed!")
    
    # Test 5: Test full update step
    print("\n✅ Test 5: Full GSPO update step")
    
    initial_params = [p.clone() for p in policy_model.parameters()]
    
    metrics = trainer.update_step(prompts)
    
    print("Training metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Check that parameters actually changed
    param_changed = False
    for initial, current in zip(initial_params, policy_model.parameters()):
        if not torch.allclose(initial, current, atol=1e-6):
            param_changed = True
            break
    
    assert param_changed, "Model parameters should have changed after update"
    assert 'loss' in metrics, "Loss should be in metrics"
    assert 'mean_reward' in metrics, "Mean reward should be in metrics"
    assert not torch.isnan(torch.tensor(metrics['loss'])), "Loss should not be NaN"
    
    print("Full GSPO update passed!")
    
    # Test 6: Test convenience function
    print("\n✅ Test 6: Convenience function")
    
    metrics2 = vectorized_gspo_update(
        policy_model, ref_model, optimizer, prompts, test_reward_fn, config
    )
    
    print("Convenience function metrics:")
    for key, value in metrics2.items():
        print(f"  {key}: {value:.4f}")
    
    assert 'loss' in metrics2, "Loss should be in metrics from convenience function"
    print("Convenience function passed!")
    
    # Test 7: Multiple update steps
    print("\n✅ Test 7: Multiple update steps")
    
    losses = []
    for step in range(3):
        metrics = trainer.update_step(prompts)
        losses.append(metrics['loss'])
        print(f"Step {step + 1}: loss = {metrics['loss']:.4f}")
    
    print(f"Loss trajectory: {[f'{l:.4f}' for l in losses]}")
    print("Multiple update steps passed!")
    
    print("\n🎉 All tests passed!")
    print("The fixed GSPO implementation is working correctly!")
    
    return metrics

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run the test
    final_metrics = run_gspo_test()
    
    print(f"\n📊 Final Training Metrics:")
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.6f}") 