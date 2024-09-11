import { ChatMistralAI } from "@langchain/mistralai";
import { ChatPromptTemplate } from "@langchain/core/prompts";

// Import environment variables
import * as dotenv from "dotenv";
dotenv.config();

// Instantiate the model
const model = new ChatMistralAI({
  modelName: "mistral-large-latest",
  temperature: 0.9,
});

// Create Prompt Template using fromTemplate
// const prompt = ChatPromptTemplate.fromTemplate('Tell a joke about {word}');

// Create Prompt Template from fromMessages
const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "You are a talented chef.  Create a recipe based on a main ingredient provided by the user.",
  ],
  ["human", "{word}"],
]);

const chain = prompt.pipe(model);

const response = await chain.invoke({
  word: "dog",
});

console.log(response);