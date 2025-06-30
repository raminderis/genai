import os
import csv

from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
INPUT_FILENAME = "D:\\tech-learning\\AI-ML\\Visionworks\\chatagent\\ping_test_results.csv"
OUTPUT_FILENAME = "D:\\tech-learning\\AI-ML\\Visionworks\\chatagent\\ping_test_results_embeddings.csv"


csvfile_in = open(INPUT_FILENAME, encoding="utf8", newline='')
input_quad = csv.DictReader(csvfile_in)

csvfile_out = open(OUTPUT_FILENAME, "w", encoding="utf8", newline='')
fieldnames = ['testtype','time_executed','sourceip','sourcenode','targetip','targetnode','jitter','throughput','rtt','uplink','downlink','resultPlot','resultPlot_embedding']
output_quad = csv.DictWriter(csvfile_out, fieldnames=fieldnames)
output_quad.writeheader()

llm = OpenAI()

for row in input_quad:
    print(row['resultPlot'])
    resultPlot = row['resultPlot'].replace('\n', ' ')
    resultPlot_response = llm.embeddings.create(
        input=resultPlot,
        model="text-embedding-ada-002"
    )
    output_quad.writerow({
        'testtype': row['testtype'],
        'time_executed': row['time_executed'],
        'sourceip': row['sourceip'],
        'sourcenode': row['sourcenode'],
        'targetip': row['targetip'],
        'targetnode': row['targetnode'],
        'jitter': row['jitter'],
        'throughput': row['throughput'],
        'rtt': row['rtt'],
        'uplink': row['uplink'],
        'downlink': row['downlink'],
        'resultPlot': row['resultPlot'],
        'resultPlot_embedding': resultPlot_response.data[0].embedding
    })

csvfile_in.close()
csvfile_out.close()