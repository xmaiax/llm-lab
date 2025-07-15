const qwen2VL_modelIdentifier = 'onnx-community/Qwen2-VL-2B-Instruct'
const device = process.env.DEVICE || 'cpu'
const dataType = process.env.DATA_TYPE ||  'q4f16'
const cacheDirectory = process.env.CACHE_DIR || './.cache'
const resizeInPixels = process.env.RESIZE_PIXELS || 512
const uploadDirectory = process.env.UPLOAD_DIR || `./upload`

const express = require('express')
const multer  = require('multer')
const fs = require('fs')

const removeFile = (filePath) => fs.unlink(filePath, (err) => {
  if(err) console.error(`Error deleting file '${filePath}':`, err)
  else console.log(`File '${filePath}' deleted.`)
})

class Qwen2VLEngine {
  static #modelFiles = null
  static #instance = null
  static #model = null
  static #rawImage = null
  static async initialize() {
    fs.readdirSync(uploadDirectory).forEach(x => removeFile(`${uploadDirectory}/${x}`))
    this.#modelFiles = {}
    const { AutoProcessor, Qwen2VLForConditionalGeneration, RawImage, env } = await import('@huggingface/transformers')
    env.cacheDir = cacheDirectory
    this.#instance = await AutoProcessor.from_pretrained(qwen2VL_modelIdentifier, {
      progress_callback: (p) => {
        if(!this.#modelFiles[p.file]) this.#modelFiles[p.file] = {}
        this.#modelFiles[p.file]['status'] = p.status
        if(!!p.progress) this.#modelFiles[p.file]['progress'] = p.progress
      },
      dtype: dataType,
      cache_dir: cacheDirectory,
      device
    })
    this.#model = await Qwen2VLForConditionalGeneration.from_pretrained(qwen2VL_modelIdentifier)
    this.#rawImage = RawImage
  }
  static checkIfReady() {
    if(!this.#instance || !this.#model || !this.#rawImage || !Object.keys(this.#modelFiles)
        .filter(key => this.#modelFiles[key].progress < 100 || this.#modelFiles[key].status !== 'done'))
      throw `Application isn't ready.`
  }
  static async execute(image, prompt, maxNewTokens) {
    this.checkIfReady()
    const resizedImage = await(await this.#rawImage.read(image)).resize(resizeInPixels, resizeInPixels)
    const inputs = await this.#instance(this.#instance.apply_chat_template([
      {
        role: 'user',
        content: [{ type: 'image' }, { type: 'text', text: prompt }]
      }], { add_generation_prompt: true }), resizedImage)
    const outputs = await this.#model.generate({...inputs, max_new_tokens: maxNewTokens })
    return this.#instance.batch_decode(outputs.slice(null,
      [inputs.input_ids.dims.at(-1), null]), { skip_special_tokens: true })[0]
  }
}

Qwen2VLEngine.initialize()

const upload = multer({ dest: `${uploadDirectory}/` })
const router = express.Router()

/**
* @swagger
* /qwen2-vl/image:
*   post:
*     tags:
*     - qwen-2-vl
*     summary: Process image with prompt.
*     operationId: image2Text
*     parameters:
*       - name: userPrompt
*         in: query
*         required: false
*         schema:
*           type: string
*           default: Descreva esta imagem.
*       - name: maxNewTokens
*         in: query
*         required: false
*         schema:
*           type: integer
*           format: int32
*           default: 128
*     requestBody:
*       content:
*         multipart/form-data:
*           schema:
*             type: object
*             required: image
*             properties:
*               image:
*                 type: string
*                 format: binary
*     responses:
*       200:
*         description: Image were processed successfully
*         content:
*           application/json:
*             schema:
*               type: object
*               description: Image2TextResponse
*               properties:
*                 upload:
*                   type: object
*                   properties:
*                     name:
*                       type: string
*                       description: Original file name
*                     mimetype:
*                       type: string
*                       description: Mime Type based on original file content
*                     size:
*                       type: integer
*                       description: Upload size in bytes
*                 output:
*                   type: string
*                   description: Model response to given prompt about the uploaded image
*       400:
*         description: Client-side error (BadRequest)
*         content:
*           application/json:
*             schema:
*               type: object
*               description: Image2TextError
*               properties:
*                 error:
*                   type: string
*                   description: Error message
*       500:
*         description: Server-side error (Internal Server Error)
*         content:
*           application/json:
*             schema:
*               type: object
*               description: Image2TextError
*               properties:
*                 error:
*                   type: string
*                   description: Error message
*/

router.post('/image', upload.single('image'), (req, res) => {
  if(!!req && !!req.query && !!req.query.userPrompt &&
    !!req.query.maxNewTokens && !!req.file) {
    console.log(`File '${req.file.originalname}' uploaded as '${req.file.path}' (size: ${req.file.size}).`)
    Qwen2VLEngine.execute(req.file.path, req.query.userPrompt, req.query.maxNewTokens).then(output => {
      removeFile(req.file.path)
      res.status(200).json({
        upload: {
          name: req.file.originalname,
          mimetype: req.file.mimetype,
          size: req.file.size
        },
        output
      })
    }).catch(reason => {
      removeFile(req.file.path)
      res.status(500).json({ error: `${reason}` })
    })
  }
  else res.status(400).json({ error: 'Invalid request.' })
})

module.exports = router
