import * as dotenv from "dotenv";
dotenv.config();

import readline from "readline";

import { ChatMistralAI } from "@langchain/mistralai";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";

import { HumanMessage, AIMessage } from "@langchain/core/messages";

import { createOpenAIFunctionsAgent, AgentExecutor } from "langchain/agents";

// Tool imports
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { createRetrieverTool } from "langchain/tools/retriever";

// Custom Data Source, Vector Stores
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { MistralAIEmbeddings } from "@langchain/mistralai";

// Create Retriever
const loader = new CheerioWebBaseLoader(
  "https://www.lycamobile.us/en/"
);
const docs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 50,
  chunkOverlap: 20,
});

const splitDocs = await splitter.splitDocuments(docs);

const embeddings = new MistralAIEmbeddings();

const vectorStore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);

const retriever = vectorStore.asRetriever({
  k: 2,
});

// Instantiate the model
const model = new ChatMistralAI({
  modelName: "mistral-large-latest",
  temperature: 0.2,
});

// Prompt Template
const prompt = ChatPromptTemplate.fromMessages([
  ("system", "You are a helpful assistant."),
  new MessagesPlaceholder("chat_history"),
  ("human", "{input}"),
  new MessagesPlaceholder("agent_scratchpad"),
]);

// Tools
const searchTool = new TavilySearchResults();
const retrieverTool = createRetrieverTool(retriever, {
  name: "lyca_search",
  description:
    "Use this tool when searching for information about Lyca",
});

const tools = [searchTool, retrieverTool];

const agent = await createOpenAIFunctionsAgent({
  llm: model,
  prompt,
  tools,
});

// Create the executor
const agentExecutor = new AgentExecutor({
  agent,
  tools,
});

// User Input

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const chat_history = [];

function askQuestion() {
  rl.question("User: ", async (input) => {
    if (input.toLowerCase() === "exit") {
      rl.close();
      return;
    }

    const response = await agentExecutor.invoke({
      input: input,
      chat_history: chat_history,
    });

    console.log("Agent: ", response.output);

    chat_history.push(new HumanMessage(input));
    chat_history.push(new AIMessage(response.output));

    askQuestion();
  });
}

askQuestion();