import torch
import time
def test_gradient_computation_cliff():
    """
    Test if the performance cliff is specifically due to 
    higher-order gradient computation through sequential parameter updates
    """
    
    device = torch.device('cuda')
    
    def create_sequential_gradient_chain(num_steps):
        """Create a chain of parameter updates similar to your MLP states"""
        
        # Initial "memory" parameters (like your K, Q, V, alpha, etc.)
        memory_params = {
            'K': torch.randn(512, 64, device=device, requires_grad=True),
            'Q': torch.randn(512, 64, device=device, requires_grad=True),
            'alpha': torch.randn(64, device=device, requires_grad=True),
        }
        
        # Initial "MLP" parameters  
        current_mlp_params = {
            'weight1': torch.randn(64, 128, device=device, requires_grad=True),
            'weight2': torch.randn(128, 64, device=device, requires_grad=True),
        }
        
        accumulated_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        for step in range(num_steps):
            # Simulate chunk input
            chunk_input = torch.randn(32, 512, device=device)
            
            # Simulate your memory interaction - make sure ALL memory params are used
            queries = chunk_input @ memory_params['Q']
            keys = chunk_input @ memory_params['K']
            
            # Simulate MLP forward pass with current params
            mlp_hidden = torch.relu(queries @ current_mlp_params['weight1'])
            mlp_output = mlp_hidden @ current_mlp_params['weight2']
            
            # Make sure alpha is used in the computation
            step_loss = (mlp_output * memory_params['alpha']).sum()
            
            # Add to accumulated loss
            accumulated_loss = accumulated_loss + step_loss
            
            # Simulate MLP parameter update (THIS IS THE KEY PART)
            # Create new MLP parameters that depend on the previous ones
            if step < num_steps - 1:  # Don't update on last step
                mlp_grads = torch.autograd.grad(step_loss, current_mlp_params.values(), 
                                              create_graph=True, retain_graph=True)
                
                # Update parameters (maintaining gradient flow)
                updated_params = {}
                for (name, param), grad in zip(current_mlp_params.items(), mlp_grads):
                    updated_params[name] = param - 0.01 * grad  # gradient step
                
                current_mlp_params = updated_params
        
        return accumulated_loss, memory_params
    
    # Test different numbers of sequential updates
    results = {}
    
    for num_steps in [1, 2, 3, 4, 5, 6, 7, 8]:
        print(f"\n=== Testing {num_steps} sequential gradient steps ===")
        
        # Clear memory
        torch.cuda.empty_cache()
        
        # Warmup
        for _ in range(3):
            try:
                test_loss, test_memory = create_sequential_gradient_chain(num_steps)
                grads = torch.autograd.grad(test_loss, test_memory.values(), 
                                          retain_graph=False, allow_unused=True)
                del test_loss, test_memory, grads
            except Exception as e:
                print(f"Warmup failed: {e}")
                break
        
        # Create the gradient chain for timing
        final_loss, memory_params = create_sequential_gradient_chain(num_steps)
        
        # Time the critical operation: computing gradients w.r.t. memory parameters
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        # This is the expensive operation in your case
        try:
            memory_gradients = torch.autograd.grad(
                final_loss, 
                memory_params.values(), 
                retain_graph=False,
                create_graph=False,
                allow_unused=True
            )
        except Exception as e:
            print(f"Gradient computation failed: {e}")
            continue
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        gradient_time = (end_time - start_time) * 1000  # milliseconds
        
        # Measure gradient computation complexity
        total_gradient_norm = sum(g.norm().item() if g is not None else 0.0 for g in memory_gradients)
        
        results[num_steps] = {
            'gradient_time_ms': gradient_time,
            'total_grad_norm': total_gradient_norm,
            'memory_allocated_mb': torch.cuda.memory_allocated() / 1024**2
        }
        
        print(f"  Gradient computation time: {gradient_time:.2f}ms")
        print(f"  Total gradient norm: {total_gradient_norm:.6f}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
        
        # Clean up
        del final_loss, memory_params, memory_gradients
        torch.cuda.empty_cache()
    
    # Analyze results
    print("\n=== RESULTS SUMMARY ===")
    print("Steps | Grad Time (ms) | Ratio vs 4-step")
    print("------|----------------|------------------")
    
    if 4 in results:
        baseline_time = results[4]['gradient_time_ms']
        for num_steps in sorted(results.keys()):
            time_ms = results[num_steps]['gradient_time_ms']
            ratio = time_ms / baseline_time if baseline_time > 0 else float('inf')
            print(f"{num_steps:5d} | {time_ms:12.2f} | {ratio:15.2f}x")
    else:
        for num_steps in sorted(results.keys()):
            time_ms = results[num_steps]['gradient_time_ms']
            print(f"{num_steps:5d} | {time_ms:12.2f} | N/A")
    
    return results

def simple_gradient_cliff_test():
    """Simplified test focusing just on sequential gradient computation"""
    print("\n=== SIMPLE GRADIENT CLIFF TEST ===")
    
    device = torch.device('cuda')
    
    for num_params in [2, 3, 4, 5, 6, 8]:
        # Create a chain of dependent parameters
        x = torch.randn(100, 100, device=device, requires_grad=True)
        
        current = x
        params = [x]
        
        for i in range(num_params):
            weight = torch.randn(100, 100, device=device, requires_grad=True)
            current = current @ weight  # Each step depends on previous
            params.append(weight)
        
        loss = current.sum()
        
        # Time gradient computation
        torch.cuda.synchronize()
        start = time.perf_counter()
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        print(f"{num_params} sequential matmuls: {(end-start)*1000:.2f}ms")
        
        # Clean up
        del x, current, params, loss, grads
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Run both tests
    print("Running detailed gradient computation cliff test...")
    detailed_results = test_gradient_computation_cliff()
    
    print("\n" + "="*50)
    simple_gradient_cliff_test()