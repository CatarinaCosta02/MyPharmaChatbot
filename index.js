import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import { exec } from "child_process"; 
import { marked } from "marked";


const app = express();
app.use(cors());

app.use(bodyParser.json());

app.post("/", (req, res) => {
    const userMessage = req.body.message;

    exec(`python script.py "${userMessage}"`, (error, stdout, stderr) => {
        if (error) {
            console.error(`Erro ao executar o script: ${error.message}`);
            return res.status(500).json({ response: "Erro ao processar a pergunta." });
        }

        if (stderr) {
            console.error(`Erro do script: ${stderr}`);
            return res.status(500).json({ response: "Erro ao processar a pergunta." });
        }

        // Converte a resposta de Markdown para HTML
        const htmlResponse = marked(stdout.trim());

        // Envia a resposta convertida como HTML para o frontend
        res.json({ response: htmlResponse });
    });
});

const port = process.env.PORT || 3001;

app.listen(port, () => {
    console.log(`Servidor em execução em http://localhost:${port}`);
});