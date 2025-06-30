import os
import csv

from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
INPUT_FILENAME = "D:\\tech-learning\\AI-ML\\Visionworks\\chatagent\\amf.csv"
OUTPUT_FILENAME = "D:\\tech-learning\\AI-ML\\Visionworks\\chatagent\\amf_embeddings.csv"


csvfile_in = open(INPUT_FILENAME, encoding="utf8", newline='')
input_quad = csv.DictReader(csvfile_in)

csvfile_out = open(OUTPUT_FILENAME, "w", encoding="utf8", newline='')
fieldnames = ['Name','Location','Technology','Market','amfPlot','amfPlot_embedding']
output_quad = csv.DictWriter(csvfile_out, fieldnames=fieldnames)
output_quad.writeheader()

llm = OpenAI()

for row in input_quad:
    print(row['amfPlot'])
    amfPlot = row['amfPlot'].replace('\n', ' ')
    amfPlot_response = llm.embeddings.create(
        input=amfPlot,
        model="text-embedding-ada-002"
    )
    output_quad.writerow({
        'Name': row['Name'],
        'Location': row['Location'],
        'Technology': row['Technology'],
        'Market': row['Market'],
        'amfPlot': row['amfPlot'],
        'amfPlot_embedding': amfPlot_response.data[0].embedding
    })

csvfile_in.close()
csvfile_out.close()