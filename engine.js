/*
q4:
onnx-community/gemma-3-1b-it-ONNX-GQA
onnx-community/Qwen2.5-0.5B-Instruct
onnx-community/Qwen3-0.6B-ONNX
onnx-community/TinySwallow-1.5B-Instruct-ONNX

q4f16:
onnx-community/Phi-3.5-mini-instruct-onnx-web

Llama <- ?
*/
const task = 'text-generation'
const model = 'onnx-community/gemma-3-1b-it-ONNX-GQA'
const cacheDirectory = './.cache'
const dataType = 'q4f16' // "auto" | "fp32" (precisão completa) | "fp16" (meia-precisão) | "q8" / "int8" / "uint8" (8-bit) | "q4" / "bnb4" / "q4f16" (4-bit)
const device = 'cpu'
const maxNewTokens = 512
const doSample = false

module.exports = class TextEngine {

  static #instance = null

  static async sendPrompt(content, progressCallback) {
    if(this.#instance === null) {
      let { pipeline, env } = await import('@huggingface/transformers')
      env.cacheDir = cacheDirectory;
      this.#instance = await pipeline(task, model, {
        progress_callback: progressCallback,
        dtype: dataType,
        cache_dir: cacheDirectory,
        device
      });
    }
    const output = await this.#instance(
      [
        { role: "system", content: "Você é um bom assistente." },
        { role: 'user', content }
      ],
      {
        max_new_tokens: maxNewTokens,
        do_sample: doSample
      }) 
    return output[0].generated_text.at(-1).content
  }

}
