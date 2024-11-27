import express from 'express';
import bodyParser from 'body-parser';
import cors from 'cors';

const app = express();
app.use(cors());

// Middleware para lidar com JSON
app.use(bodyParser.json());

/*
app.post('/', (req, res) => {
  //  const userMessage = req.body.message;
    console.log('Mensagem recebida do frontend:', req.body.message);
    res.send({ response: 'Mensagem processada com sucesso!' });

   // const chatbotResponse = `Você disse: "${userMessage}". Aqui está minha resposta!`;
    // Envia a resposta para o frontend
    res.send({ response: chatbotResponse });
});
*/
const { exec } = require("child_process");

app.post("/", (req, res) => {
    const userMessage = req.body.message;

    exec(`python3 run_chatbot.py "${userMessage}"`, (error, stdout, stderr) => {
        if (error) {
            console.error(`Erro ao executar o script: ${error.message}`);
            return res.status(500).json({ response: "Erro ao processar a pergunta." });
        }

        if (stderr) {
            console.error(`Erro do script: ${stderr}`);
            return res.status(500).json({ response: "Erro ao processar a pergunta." });
        }

        // Envia a resposta 
        res.json({ response: stdout.trim() });
    });
});

const port = process.env.PORT || 3001;

app.listen(port, () => {
    console.log(`Serve at http://localhost:${port}`);
});
