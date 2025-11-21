import pyarrow.parquet as pq
import pandas as pd
def get_data(path,n=1000,threshold=20):
  data=[]
  parquet=pq.ParquetFile(path)
  for batch in parquet.iter_batches(batch_size=1000,columns=["sentence"]):
    df_batch=batch.to_pandas()
    for sentence in df_batch["sentence"]:
      if sentence.endswith("."):
        sentence=sentence[:-1]
      tokens=sentence.split()
      safe=True
      for token in tokens:
        if len(token)>=threshold:
          safe=False
          break
        
      if safe:
        data.append(sentence)
      if len(data)==n:
        return data
  return data