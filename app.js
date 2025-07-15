const swaggerUi = require('swagger-ui-express')
const hostname = require('os').hostname()
const swaggerSuffix = '/swagger'

const app = require('express')()
const port = process.env.PORT || 3000

app.use(swaggerSuffix, swaggerUi.serve, swaggerUi.setup(require('swagger-jsdoc')({
    swaggerDefinition: {
        openapi: '3.0.0',
        info: {
            title: 'Qwen2-VL',
            version: 'mk-i'
        },
        servers: [{ url: `/` }]
    },
    apis: ['./routes/*.js'],
})))

app.use('/qwen2-vl', require('./routes/qwen2-controller'))

app.listen(port, () => console.log(`Running: http://${hostname}:${port}${swaggerSuffix}`))
