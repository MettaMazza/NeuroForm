import logging
import os
from dotenv import load_dotenv
load_dotenv()

from neuroform.memory.graph import KnowledgeGraph
from neuroform.memory.neuroplasticity import AutonomousNeuroplasticity
from neuroform.llm.ollama_client import OllamaClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def main(): # pragma: no cover
    print("====== NeuroForm Standalone Boot ======")
    kg = KnowledgeGraph()
    if not kg.driver:
        print("Neo4j not connected. Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD.")
        return

    model = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
    neuro = AutonomousNeuroplasticity(kg, model=model)
    llm = OllamaClient(kg, model=model)

    print("\\n[Running Autonomous Memory Optimization]")
    opt_result = neuro.evaluate_and_optimize()
    print(f"Status: {opt_result['status']}, Actions: {opt_result['actions_taken']}")

    print("\\n[NeuroForm is online. Type 'exit' to quit.]")
    while True:
        try:
            msg = input("\\nUser: ")
            if msg.lower() == "exit":
                break
            
            response = llm.chat_with_memory("local_user", msg)
            print(f"\\nNeuroForm: {response}")
            
        except KeyboardInterrupt:
            break

    kg.close()
    print("\\nShutdown complete.")

if __name__ == "__main__": # pragma: no cover
    main()
