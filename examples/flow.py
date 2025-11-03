"""
Tutorial #3: Flow in JAX and PCX

This script demonstrates how to use control flow with JAX and PCX transformations,
including cond, scan, and while_loop.
"""

import jax
import jax.numpy as jnp
import numpy as np

import pcx as px
import pcx.functional as pxf


def main():
    print("=" * 80)
    print("Tutorial #3: Flow in JAX and PCX")
    print("=" * 80)
    
    # ==============================================================================
    # Example 1: Static Flow with Python
    # ==============================================================================
    
    print("\n--- Example 1: Static Flow (Conditional on Static Values) ---\n")
    
    model = {
        'x': px.Param(1.0),
        'c': px.static(True)
    }
    
    @pxf.jit()
    def f(*, model):
        print("f is being compiled...")
        
        # Can use Python if/else with static values
        if model['c'].get():
            model['x'] += 1.0
        else:
            model['x'] -= 1.0
    
    print("Calling f with c=True:")
    f(model=model)
    print(f"x: {model['x'].get()}")
    f(model=model)
    print(f"x: {model['x'].get()}")
    
    print("\nChanging c to False (triggers recompilation):")
    model['c'].set(False)
    f(model=model)
    print(f"x: {model['x'].get()}")
    f(model=model)
    print(f"x: {model['x'].get()}")
    
    # ==============================================================================
    # Example 2: Dynamic Flow with cond
    # ==============================================================================
    
    print("\n--- Example 2: Dynamic Flow with cond ---\n")
    
    def choice_a(x: jax.Array, *, p: px.Param):
        """Branch A: subtract x from p."""
        p -= x
    
    def choice_b(x: jax.Array, *, p: px.Param):
        """Branch B: multiply p by x."""
        p.set(p * x)
    
    @pxf.jit()
    def f_cond(x: jax.Array, c: bool, *, p: px.Param):
        """Conditional execution based on dynamic value c."""
        pxf.cond(choice_a, choice_b)(c, x, p=p)
    
    param = px.Param(jnp.array([1.0]))
    x = jnp.array([-2.0])
    
    print("Initial p:", param.get().item())
    
    print("Calling with c=True (choice_a: p = p - x):")
    f_cond(x, True, p=param)  # 1.0 - (-2.0) = 3.0
    print(f"p: {param.get().item()}")
    
    print("Calling with c=False (choice_b: p = p * x):")
    f_cond(x, False, p=param)  # 3.0 * (-2.0) = -6.0
    print(f"p: {param.get().item()}")
    
    assert param.get().item() == -6.0, "Result should be -6.0"
    
    # ==============================================================================
    # Example 3: Loops with scan
    # ==============================================================================
    
    print("\n--- Example 3: Loops with scan ---\n")
    
    @pxf.jit()
    def fix_many_f(x: jax.Array, c: jax.Array, *, p: px.Param):
        """Apply operation multiple times based on array c."""
        def f(i, x, *, p):
            pxf.cond(choice_a, choice_b)(i, x, p=p)
            
            # scan requires returning (args, output)
            return x, None
        
        pxf.scan(f, c)(x, p=p)
    
    param = px.Param(jnp.array([1.0]))
    x = jnp.array([-2.0])
    
    # Sequence of operations: False, False, True, False, True, True, False, True
    c = jnp.array([False, False, True, False, True, True, False, True])
    
    print("Initial p:", param.get().item())
    print(f"Applying sequence: {c}")
    
    fix_many_f(x, c, p=param)
    print(f"Final p: {param.get().item()}")
    
    assert param.get().item() == 18.0, "Result should be 18.0"
    
    # ==============================================================================
    # Example 4: While Loop
    # ==============================================================================
    
    print("\n--- Example 4: While Loop ---\n")
    
    @pxf.jit()
    def var_many_f(x: jax.Array, *, p: px.Param):
        """Run loop until condition is met (or max iterations)."""
        def f(x, count, *, p):
            # Randomly choose operation
            c = jax.random.bernoulli(px.RKG())
            pxf.cond(choice_a, choice_b)(c, x, p=p)
            
            return x, count + 1
        
        def loop_cond(x, count, *, p):
            """Continue while p > 0 and count < 3."""
            return jnp.all(jnp.logical_and(p > 0.0, count < 3))
        
        return pxf.while_loop(f, loop_cond)(x, 0, p=p)
    
    # Test multiple times to show randomness
    print("Running while loop 5 times with random operations:")
    px.RKG.seed(42)
    
    for i in range(5):
        param = px.Param(jnp.array([1.0]))
        x = jnp.array([-2.0])
        
        _, count = var_many_f(x, p=param)
        print(f"  Run {i+1}: p={param.get().item():6.2f}, steps={count}")
    
    # Statistical test
    print("\nStatistical test (1024 runs):")
    px.RKG.seed(42)
    values = []
    for i in range(1024):
        param = px.Param(jnp.array([1.0]))
        var_many_f(x, p=param)
        values.append(param.get().item() > 0)
    
    # Each iteration has 50% chance to go negative, so ~1/8 should stay positive
    positive_fraction = np.mean(values)
    print(f"Fraction positive: {positive_fraction:.3f} (expected ~0.125)")
    
    assert np.abs(positive_fraction - 1/8) < 0.05, "Should be close to 1/8"
    
    # ==============================================================================
    # Summary
    # ==============================================================================
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("\nKey points about control flow in JAX/PCX:")
    print("  1. Python if/else works only with STATIC values")
    print("  2. Use pxf.cond for dynamic conditionals")
    print("  3. Use pxf.scan for fixed-length loops")
    print("  4. Use pxf.while_loop for condition-based loops")
    print("  5. All transformations automatically track parameters")
    print("\nThese primitives enable complex control flow in compiled code!")


if __name__ == "__main__":
    main()