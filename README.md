## Lamma2_7b finetuned on custom data

This project aims to fine-tune the Llama-2 language model using Hugging Face's Transformers library. By following these steps, you can fine-tune the model and use it for inference.

### Setup

1. Clone this repository to your local machine.

2. Create a `.env` file in the project directory and add your Hugging Face API token:
   ```HUGGING_FACE_API_KEY = "your_HF_API_key"```<br>
   The code for training (train.py) has the code to pick this API key up.<br>


3. Install the required packages using the following command:

   ```bash
   pip install -r requirements.txt
   ```

4. Train llm with reference to ->  [notebook](https://github.com/Sagarkeshave/finetune_llama2_7b/blob/main/src/finetune_llama2_7b__main.ipynb)

5. Custom Data Ingestion
To ingest your own data for fine-tuning, you'll need to modify the code in your script. I have provided one example here:

```python
#Reading the file
data = pd.read_excel("your_dataset.xlsx")

# Convert the pandas DataFrame to Hugging Face's Dataset
hf_dataset = Dataset.from_pandas(data)

```
## Inference

To perform inference using the fine-tuned Llama-2 model, follow these steps:

1. Ensure you've successfully fine-tuned Llama-2 as explained above

2. Run the inference script, `infer.py`, with the following command:

   ```shell
   !python infer.py
   ```


## Training procedure

The following `bitsandbytes` quantization config was used during training:
- load_in_8bit: False
- load_in_4bit: True
- llm_int8_threshold: 6.0
- llm_int8_skip_modules: None
- llm_int8_enable_fp32_cpu_offload: False
- llm_int8_has_fp16_weight: False
- bnb_4bit_quant_type: nf4
- bnb_4bit_use_double_quant: False
- bnb_4bit_compute_dtype: float16
  
### Framework versions
- PEFT 0.4.0
