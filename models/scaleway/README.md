## Overview

[Scaleway Generative APIs](https://www.scaleway.com/en/generative-apis/) serve open-weight models (Qwen, DeepSeek, Mistral, Gemma, GLM, Llama, gpt-oss, Whisper…) from data centers in Paris, France, through an OpenAI-compatible API. This plugin makes them available in Dify as a model provider.

Supported model types:

- **LLM**: chat completion with streaming, tool calling, structured output (`json_object` / `json_schema`), vision on multimodal models, and `reasoning_effort` on reasoning models.
- **Text Embedding**: `qwen3-embedding-8b`, `bge-multilingual-gemma2`.
- **Speech to Text**: `whisper-large-v3`.

## Configure

1. In the [Scaleway console](https://console.scaleway.com/iam/api-keys), create an API key and copy its **secret key**.
2. In Dify, go to **Settings → Model Provider → Scaleway** and enter:
   - **API Secret Key**: the secret key from step 1.
   - **Project ID** (optional): bills usage to that Project. Requests then go to `https://api.scaleway.ai/<project_id>/v1`. If you leave it empty, requests go to `https://api.scaleway.ai/v1` and are billed to the default Project.
3. Save. The plugin validates the key with a short request to `mistral-small-3.2-24b-instruct-2506`.

### Custom models

Use **Add Model** to register a serverless model that this plugin doesn't predefine yet, such as a model Scaleway released recently. Use the model ID from the [supported models](https://www.scaleway.com/en/docs/generative-apis/reference-content/supported-models/) page. Requests always go to the serverless endpoint (`api.scaleway.ai`). Dedicated Deployments are not supported.

## Updating the model list

The model YAML files are generated. Don't edit them by hand. `scripts/sync_catalog.py` reads Scaleway's public product catalog (no API key needed). The catalog provides:

- which serverless models are active (`preview` or `general_availability`);
- their EUR prices;
- whether they accept images;
- their supported reasoning efforts.

The catalog has no context window or output limit, so the script takes those from `scripts/model_profiles.json`. That file is maintained by hand from the [supported models](https://www.scaleway.com/en/docs/generative-apis/reference-content/supported-models/) page, and its key order sets the order of models in Dify.

```bash
python scripts/sync_catalog.py          # rewrite models/**/*.yaml from the live catalog
python scripts/sync_catalog.py --check  # exit 1 if the files are out of date
```

The script fails in two cases:

- **A new model is in the catalog.** Add it to `model_profiles.json` and run the script again.
- **A profiled model was retired.** Remove it from `model_profiles.json` and run the script again.

## Notes

- The predefined models are the serverless models listed in Scaleway's [supported models](https://www.scaleway.com/en/docs/generative-apis/reference-content/supported-models/) page that are not deprecated. Models that Scaleway has retired are not included.
- Token prices are in EUR per million tokens, taken from [Scaleway pricing](https://www.scaleway.com/en/pricing/model-as-a-service/) (November 2026, Paris region, before tax). Dify's cost estimates don't include the cached-input discount, the -50% Batches API discount or the free tier. `whisper-large-v3` is billed per audio minute (€0.003), which Dify doesn't track.
