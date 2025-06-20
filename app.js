const input = `faça um poema sobre machine learning`

const consoleLog = (x) => console.log(x)
const TextEngine = require('./engine.js')
TextEngine.sendPrompt(input, consoleLog).then(consoleLog)
