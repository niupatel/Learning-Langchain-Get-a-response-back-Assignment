import { createAgent, tool, type ToolRuntime } from "langchain";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai"; 
import { MemorySaver } from "@langchain/langgraph"; 
import * as z from "zod"; 

// Define system prompt.
const systemPrompt = `You are a concise weather assistant.
You have access to two tools:
- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location
If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location.`; // System instructions for the model.

// Define tools.
const getWeather = tool(
  ({ city }) => `It's always sunny in ${city}!`, 
  {
    name: "get_weather_for_location", 
    description: "Get the weather for a given city", 
    schema: z.object({
      city: z.string(), 
    }), 
  }
); 

type AgentRuntime = ToolRuntime<unknown, { user_id: string }>; 

const getUserLocation = tool(
  (_, config: AgentRuntime) => {
    const { user_id } = config.context;
    return user_id === "1" ? "Santa Cruz, Brazil" : "SF"; 
  },
  {
    name: "get_user_location", 
    description: "Retrieve user information based on user ID", 
    schema: z.object({}), 
  }
); 
async function main() {
  const model = new ChatGoogleGenerativeAI({
    model: "gemini-2.5-flash", 
    temperature: 0, 
  }); 

  // Define response format.
  const responseFormat = z.object({
    weather_summary: z.string().optional(), 
    good_for_sport: z.boolean().optional(), 
    sport: z.string().optional(), 
    best_brazilian_athlete: z.string().optional(), 
  }); 

  // Set up memory.
  const checkpointer = new MemorySaver(); 

  // Create agent.
  const agent = createAgent({
    model,
    systemPrompt, 
    responseFormat,
    checkpointer, 
    tools: [getUserLocation, getWeather], 
  }); 

  // Run agent.
  // `thread_id` is a unique identifier for a given conversation.
  const config = {
    configurable: { thread_id: "1" }, 
    context: { user_id: "1" }, 
  }; 
  const response = await agent.invoke(
    {
      messages: [
        {
          role: "user",
          content:
            "What's the weather in Santa Cruz, Brazil? Reply with weather_summary only.",
        },
      ],
    }, // User message.
    config // Thread + context.
  ); // End invoke.
  const weatherSummary = response.structuredResponse.weather_summary; 

  //  use the weather summary to ask what sport is good to play.
  const sportPrompt = `Based on this weather, what sport should I play? Weather: ${weatherSummary}. Reply with sport only.`; 
  const sportResponse = await agent.invoke(
    { messages: [{ role: "user", content: sportPrompt }] },
    config // Same thread + context.
  ); // End second invoke.
  const sport = sportResponse.structuredResponse.sport; 

  // ask for a Brazilian athlete.
  const athletePrompt = `Name a top Brazilian athlete for this sport: ${sport}. Reply with best_brazilian_athlete only.`; 
  const athleteResponse = await agent.invoke(
    { messages: [{ role: "user", content: athletePrompt }] }, 
    config 
  );
  const athlete = athleteResponse.structuredResponse.best_brazilian_athlete; 

  // Print exactly three sentences.
  console.log(
    `The weather is ${weatherSummary}. A good sport to play is ${sport}. A top Brazilian athlete in ${sport} is ${athlete}.`
  );
} 

main().catch(console.error); // Run main and log any errors.
