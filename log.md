
## Transformer

### Self-attention
Input embeddings $\{x_i\}_{i=1}^{n}$
Output embeddings $\{z_i\}_{i=1}^{n}$, preserving the input dimensions

The $i$-th token is mapped via linear transformations to a key $k_i$, query $q_i$ and value $v_i$.

$$
z_i=\sum_{j=1}^{n}softmax(\{\langle q_i,k_{j'} \rangle\}_{j'=1}^{n}) \cdot v_j.
$$

### Casual self-attention

Enable autoregressive generation

$$
z_i=\sum_{j=1}^{i}softmax(\{\langle q_i,k_{j'} \rangle\}_{j'=1}^{i}) \cdot v_j.
$$

Multi-head attention

Cross-attention



## Pretrain

| Parameter                         | Value              |
| --------------------------------- | ------------------ |
| experiment_name                   | pretrain           |
| seed                              | 42,43,44           |
| env_params                        | multi_domain_mtdmc |
| run_params                        | pretrain           |
| eval_params                       | pretrain_disc      |
| agent_params                      | cdt_pretrain_disc  |
| agent_params.kind                 | MDDT               |
| agent_params/model_kwargs         | multi_domain_mtdmc |
| agent_params/data_paths           | mt40v2_dmc10       |
| agent_params/replay_buffer_kwargs | multi_domain_mtdmc |
| agent_params/accumulation_steps   | 2                  |

MDDT: Multi-domain Decision Transformer

- act_dim: 4
- max_act_dim: 6
- state_dim: 204

embed_state: Linear 204 -> 768
embed_rewards: Linear 1 -> 768
embed_return: Linear 1 -> 768
embed_action_disc: Embedding 65embeds with dim 768, the 64-th is padding embed

Huggingface parameters:

- max_length: 5. The max length of the sequence to be generated.
- n_embed: 512. 
- n_layer: 6. Number of hidden layers in the Transformer encoder.
- n_head: 12. Number of attention heads for each attention layer in the Transformer encoder.
- hidden_size: 768. The hidden size of the model
- max_ep_len: 1000

Trajectory Dataset
context length: 5
batch_size: 256

- s: [256,5,204]
- a: [256,5,6]
- s1: [256,5,204]
- r: [256,5,1]
- togo: [256,5,1]
- t: [256,5]
- mask: [256,5]
- done: [256,5]
- task_id: [256]
- trj_id: [256]
- action_mask: [256,5,6]

One difference between rl and cv is the action has multiple tokens.