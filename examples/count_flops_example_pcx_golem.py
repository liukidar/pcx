#################### THIS SNIPPETS NEEDS TO BE PLACED RIGHT AFTER THE MODEL IS TRAINED ####################

def count_flops_with_real_batch(model, real_dl):
    # 1) Just pick the first batch from your real data loader:
    for x_batch in dl:
        break # shape is (batch_size=32, 2), same as used in training

    # 2) Define a single-pass function that calls your actual forward
    @jax.jit
    def single_pass(x_batch):

        # We do a single forward pass inside a pxu.step block
        with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
            # exactly the same call you do in eval_on_batch, but returning raw logits
            forward(x_batch, model=model)

            return forward(None, model=model)

    # 3) Compile that single-pass function and estimate FLOPs
    c = single_pass.lower(x_batch).compile()
    cost_analysis = c.cost_analysis()
    print(cost_analysis)   # [{'bytes accessed2{}': 8192.0, 'optimal_seconds': -3.0, 'bytes accessed': 17157.0, 'utilization1{}': 6.0, 'utilization0{}': 9.0, 'bytes accessed0{}': 8445.0, 'bytes accessed1{}': 261.0, 'flops': 8253.0, 'utilization2{}': 2.0, 'bytes accessedout{}': 8445.0}]
    print(cost_analysis[0])
    flops = cost_analysis[0]["flops"]
    return flops

flops = count_flops_with_real_batch(model, dl)
print(f"Estimated FLOPs using a real batch from train_dl: {flops:,}")