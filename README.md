# inverse_agent

LLM agent that recovers thermal diffusivity α from sparse temperature observations by iteratively querying a black-box MSE oracle w/o gradients or optimizer.

CoMSAIL Lab, Cornell University. [[Paper]](https://drive.google.com/file/d/1HIOF-HLAIsM-MNfiDT4pAjEVxhBXru6J/view?usp=sharing)

## Setup

```bash
pip install groq mcp jax jaxlib python-dotenv matplotlib
```

```
GROQ_API_KEY=your_key_here
```

## Usage

```bash
# single trial
HEAT_TRUE_ALPHA=0.006 HEAT_T_FINAL=5.0 python mcp_client.py

# automated sweep (resumes if interrupted)
python run_trials.py
```

## How it works

The agent calls `evaluate_mse(alpha)`, gets back a scalar MSE, and refines its guess. The MCP server runs the JAX implicit solver and computes MSE at the configured sensor locations.

Change sensor setup in `mcp_server.py`:

```python
sensor_indices = jnp.array([N // 5])          # single sensor at x = -0.6
sensor_indices = jnp.arange(N)                # full domain
```

## Key findings

- More sensors consistently degrades performance (75% → 26% as sensors increase 1 → 5)
- Search strategy, not observability, is the binding constraint
- α = 0.006 fails at 0% across all configurations due to bisection not reaching non-dyadic values

## Stack

- **LLM:** Llama-4-Scout via Groq
- **Tool interface:** Model Context Protocol (MCP)  
- **Solver:** JAX backward Euler
