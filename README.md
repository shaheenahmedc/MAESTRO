# Multi-Agent SWE-bench - AISI Bounty Project 

This repo contains the final report and code for the MAESTRO scaffolding bounty project with AISI. The entrypoint to inspect_evals' SWE-Bench [implementation](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/swe_bench) is via the solvers in `swe_bench.py`, where a single agent and (single agent with multi-agent tool) solver can be found. 

The `consult_multi_agent_team()` tool used [here](https://github.com/shaheenahmedc/MAESTRO/blob/c9da7af28365e6df7e7fd858c927418dd1a3932c/swe_bench.py#L231) should be usable across all Inspect solvers. 

Todo: remove swe_bench task code. 

## Overview

Within this project, we sought to investigate whether we could elicit capability uplifts on SWE-Bench, via the use of multi-agent systems. We wanted to measure performance against single-agent baselines, normalising for tokens across experiments. A copy of the accompanying report can be found in this repo.

Within this PR, you will find the following files of interest:
- `autogen_team/` - this folder contains all code required to create an Autogen Core multi-agent team, which is then wrapped in an Inspect subtask, and finally an Inspect tool. 
- `swe_agent_tools/` - this folder contains a set of swe-agent-style tools (developed at AISI), with the addition of a `run_bash_command` tool, the AISI-internal version of which wasn't shareable. 
- `we_bench.py` - this file contains minor modifications to the original SWE-bench implementation, to allow for the use of the multi-agent tool. 
- `extract_tokens.py` - this file allows us to extract the number of tokens the multi-agent system uses across each of the SWE-Bench tasks, and provides us with the mean and standard deviation of the number of tokens used. 
- `sample_analysis.py` - this file allows us to see, for a number of Inspect output logs, which runs had the same solved samples. This allowed us to quickly identify trajectories of interest, for instance where a multi-agent was able to solve a task, but all single-agent baselines were not. 

Within the `autogen_team/` folder, you will find the following files and folders:
- `agents/` - this folder contains files defining the `aggregator` and `consultant` agents, which form the multi-agent team. 
- `configs/` - this folder contains the config files for each of the multi-agent systems we used. We're able to set the `experiment_name` and `log_base_path`, which controls the folder in which the multi-agent logs are stored. We're also able to tune the number of reflection steps from here. We then define a number of agents, each with their own model, provider, and a number of other parameters. These are passed to Autogen's OpenAI chat completion client, and subsequently to OpenRouter's API, when we set `provider` to `openrouter`. See `models/client_factory.py` in this folder for more details. Note: we set the config in `runtime.py`, we didn't have time to facilitate setting config files from the command line. 
- `data_models.py` - this file contains some dataclasses we use in the multi-agent system. 
- `models/` - this folder contains a `client_factory.py` file, which contains the code to create the Autogen OpenAI chat completion client, which is able to use the `openrouter` provider. We use this provider to access a variety of models for our multi-agent system experiments. We also have a `token_usage.py` file, which we use to log the token usage of our multi-agent system in the output logs. 
- `utils/logging.py` - this file contains the code to log the output of our multi-agent system. We didn't see Autogen logging working with Inspect, and Python logging didn't seem to work either. We therefore created our own logging system, which logs a number of events, per agent, aggregator and team run. 
- `runtime.py` - this file contains Autogen Core code to setup the multi-agent system, using the consultant and aggregator agents defined above. The `run_team` function is the main function, which is called by the single agent in `main.py`.
- `main.py` - this file contains an Inspect subtask calling our multi-agent system, and an Inspect tool wrapping the subtask. We tried a few different ways to provide a bridged Autogen Core multi-agent Solver to an Inspect tool, but weren't successful in doing this. We opted to discard the use of the bridge feature, and call the `run_team` function directly from the subtask. 

## How to Use 
- One should be able to generally use this multi-agent tool with any Inspect Solver, as seen in `swe_bench.py`. The final answers from the aggregator agent consistently appear in the Inspect log viewer, but the intermediate results from the multi-agent system are only visible in the output logs.
- The format of the configuration files is visible in `autogen_team/configs/`.  We were able to use the direct OpenAI API via Autogen's OpenAI chat client. We were also able to access the OpenRouter API via a base URL parameter to the Autogen OpenAI chat client. See `models/client_factory.py` for more details.
- The `extract_tokens.py` can be called directly from the command line, with an argument for the folder path set in the config file via `experiment_name` appended to `log_base_path`. 
- The `sample_analysis.py` can be called directly from the command line, with arguments for each Inspect log filepath we'd like to compare. 
- We've copied the packages in our environment to `autogen_team/requirements_MAS.txt`. Note, this was just a call to `pip freeze`, so isn't a minimal set of dependencies. But the versions of all key packages (Autogen, Inspect, etc) are available. 

## Developer Notes/Future Work
- A major issue we faced was around getting successful API returns when calling Autogen's OpenAI chat completion client's `create()` method, with an OpenRouter endpoint. OpenRouter implements load balancing across multiple endpoints, which made it challenging to get consistently successful function calling. We tried to get round this with the `require_full_parameter_support` parameter, which passes the `require_parameters` parameter to the OpenRouter API. See link [here](https://openrouter.ai/docs/features/provider-routing). It seems Autogen's `create()` method only returns a `NoneType` error when the API call fails, so we found it helpful to add additional debugging outputs to Autogen's `create()` method. One of our primary hypotheses, was that we should be able to increase the diversity of thought amongst our multi-agent systems, by using more base model families. Hence we thought it worthwhile to try and get this working. 
- It seems one `NoneType` error remains, namely around contexts being passed to models which exceed their context length. We haven't found a fix for this yet, as OpenRouter's `middle-out` transform doesn't seem to work consistently.
- Performance may increase by passing the original prompt directly to the multi-agent system, rather than asking the single agent to do this. We didn't have time to implement this. 
- OpenAI, Claude and Llama models seem to work reliably via OpenRouter. Gemini models proved more difficult. We thought OpenRouter would allow us to use a single function-calling format for our API calls, but this doesn't seem to be the case.
- Occasionally, a consultant agent will skip the last round of reflection, and just immediately return an answer. Could be some sort of race condition. 
- A useful extension would be to further generalise the code, such that a user can pass any number of agents and messaging pattern in the config file. But this hasn't been implemented yet.

