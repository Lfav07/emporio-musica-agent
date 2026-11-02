# Empório da Música — agente de atendimento

Agente conversacional em português para apoiar o atendimento da loja fictícia Empório da Música. Responde perguntas sobre catálogo, preços, disponibilidade, pedidos, dados institucionais e políticas usando as fontes fornecidas no desafio técnico da Artefact.

## Requisitos

- Python 3.13 ou superior
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) em execução local
- Modelo de chat `qwen3.5:9b` e modelo de embeddings `nomic-embed-text`

## Configuração

Na pasta `emporio-musica-agent/`, instale as dependências:

```powershell
uv sync
```

Baixe os modelos (uma vez):

```powershell
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
```

O padrão é `http://localhost:11434` e `qwen3.5:9b`. Para alterar, defina `OLLAMA_BASE_URL` e/ou `OLLAMA_MODEL` no ambiente ou em um arquivo `.env` na raiz do projeto:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.5:9b
```

## Indexação das políticas

O manual `data/políticas_da_loja.pdf` é extraído, limpo, dividido em trechos com sobreposição e armazenado localmente no Chroma. Gere ou atualize o índice antes de iniciar a API:

```powershell
uv run python scripts/ingest_policies.py
```

O script realiza verificações de recuperação para regras importantes. A primeira indexação pode levar alguns minutos, pois os embeddings são calculados localmente. O banco fica em `data/chroma/`.

## Executar a API

```powershell
uv run uvicorn emporio_musica_agent.main:app --host 127.0.0.1 --port 8000
```

Documentação interativa: <http://127.0.0.1:8000/docs>. Verificação de disponibilidade: `GET /health`.

Exemplo de conversa:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/chat `
  -ContentType 'application/json' `
  -Body '{"session_id":"cliente-001","message":"Quais opções de violões disponíveis custando até R$1000?"}'
```

Cada chamada precisa de `session_id` e `message`. Mensagens com o mesmo identificador compartilham histórico em memória durante a vida do processo. A API devolve `{ "answer": "..." }`.

Para usar o chat no em modo CLI no terminal, em vez da API:

```powershell
uv run emporio-musica-agent
```

## Decisões técnicas

### Framework(s) e abordagem do agente

Usei LangChain `create_agent` para coordenar chamadas de ferramentas (function calling) em um fluxo ReAct: o modelo interpreta a pergunta, escolhe as fontes necessárias e redige a resposta com base nos resultados. A arquitetura é híbrida: consultas estruturadas aos CSVs atendem catálogo e pedidos, enquanto RAG recupera trechos do PDF para políticas e informações institucionais. Perguntas que dependem de prazo também podem chamar a ferramenta de data atual.

### Modelo e provedor

O modelo de conversa é Qwen 3.5 9B executado localmente pelo Ollama, com temperatura zero. Essa escolha dispensa chave de API e evita enviar conversas a um provedor externo, ao custo de depender da memória e capacidade de processamento da máquina. Para embeddings do RAG, uso `nomic-embed-text`, também via Ollama.

### Interface de interação

A interface principal é uma API HTTP em FastAPI: `POST /chat` recebe `session_id` e `message`, e `GET /health` permite verificar o serviço. O FastAPI também expõe documentação OpenAPI em `/docs`. Há um CLI opcional, que reutiliza o agente e permite conversar pelo terminal.

### Persistência do histórico de conversa

O histórico fica em memória, separado por `session_id`, com até 12 mensagens finais (seis turnos) por sessão. Essa opção mantém contexto durante uma conversa sem acrescentar infraestrutura de banco de dados ao protótipo. O histórico é perdido quando o processo reinicia e não é compartilhado entre instâncias; persistência durável e sincronização seriam etapas para um uso multiusuário em produção.

### Tratamento dos dados

Pandas carrega os CSVs operacionais. A busca considera produtos ativos com estoque positivo, combina nome, descrição e categoria para localizar itens, aplica filtros de categoria e preço e calcula o preço promocional com base nas promoções ativas. Pedidos são associados aos respectivos itens e os status são apresentados em português.

Para as políticas, o PDF passa por extração e limpeza de texto, é dividido em trechos de 1.200 caracteres com sobreposição de 200 e indexado localmente no Chroma. A busca semântica recupera trechos relevantes para o agente; o script de ingestão inclui verificações para regras importantes.

Uma ferramenta retorna a data atual no fuso `America/Sao_Paulo` para perguntas que exigem comparação de prazos. Na política de arrependimento, o prazo é de sete dias corridos após o recebimento; o agente não presume esse marco quando a data de recebimento não foi informada. O prompt orienta o agente a consultar ferramentas para fatos da loja, não inventar informações e usar uma resposta de fallback quando as fontes forem insuficientes.
## Exemplos de interação

Os cinco registros em [`examples/`](examples/) documentam conversas feitas pela API, com os pedidos e respostas retornadas pelo agente. Incluem catálogo, preço, informação institucional, pedido e aplicação da política de arrependimento.

## Limitações e próximos passos

- Dados e catálogo são arquivos estáticos; não há conexão a estoque, pedidos ou pagamentos de um sistema de produção.
- O histórico é perdido ao reiniciar e as sessões não são persistidas nem sincronizadas para acesso concorrente.
- A qualidade do RAG depende da extração do PDF, do modelo local e da recuperação semântica. As respostas não substituem análise humana em casos ambíguos.
- Na execução registrada para o catálogo, a resposta também mencionou um ukulele como falso positivo. Um próximo ajuste é exigir o filtro estruturado de categoria e limitar a resposta aos registros retornados por esse filtro.
- Não há autenticação, autorização por cliente, observabilidade, avaliação automatizada de respostas ou implantação em produção.
- Com mais tempo: integrar fontes transacionais reais com controles de acesso, persistir sessões, criar conjunto de avaliações para roteamento/recuperação/políticas e adicionar monitoramento e testes de carga.

## Uso de assistentes de código

Usei o Codex como assistente de código para apoiar a leitura do enunciado e do repositório, estruturar esta documentação e executar as verificações e chamadas à API. As decisões e alterações foram revisadas no contexto do código e dos dados fornecidos.