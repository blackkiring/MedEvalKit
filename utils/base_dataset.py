import torch
import os
import json
import hashlib

from tqdm import tqdm
import gc

from .utils import save_json

class BaseDataset:
  def __init__(self):
    self.chunk_idx = int(os.environ.get("chunk_idx",0))
    self.num_chunks = int(os.environ.get("num_chunks",1))

  def run(self,samples,model,batch_size = 2000):
    out_samples = []
    with torch.no_grad():
        messages_list = []
        current_messages = []
        current_samples = []
        for sample in tqdm(samples):
            messages = sample["messages"]
            current_messages.append(messages)
            current_samples.append(sample)
            if len(current_messages) >= batch_size:
                messages_list.append([current_messages,current_samples])
                current_messages = []
                current_samples = []
        if current_messages:
            messages_list.append([current_messages,current_samples])
        
        for current_messages,current_samples in tqdm(messages_list):
            outputs = model.generate_outputs(current_messages)
            try:
                for sample,response in zip(current_samples,outputs):
                    del sample["messages"]
                    sample["response"] = response
                    out_samples.append(sample)   
            except Exception as e:
                from pdb import set_trace;set_trace()
                print(e)
            gc.collect()
    return out_samples

  def cal_matrics(self):
    pass

  def init_dataset(self):
    pass

  def construct_messages(self):
    pass

  def _load_existing_results(self, results_path):
      """Load existing results from file if it exists and is valid."""
      if not os.path.exists(results_path):
          return None
      
      try:
          with open(results_path, "r") as f:
              existing_results = json.load(f)
          if isinstance(existing_results, list) and len(existing_results) > 0:
              print(f"Found existing results with {len(existing_results)} samples at {results_path}")
              return existing_results
      except (json.JSONDecodeError, IOError) as e:
          print(f"Warning: Could not load existing results from {results_path}: {e}")
          return None
      
      return None
  
  def _get_sample_key(self, sample):
      """Generate a unique key for a sample to identify duplicates."""
      # Try common unique identifiers
      if "id" in sample:
          return f"id:{sample['id']}"
      if "question" in sample and "answer" in sample:
          # Use stable hash that's consistent across restarts
          content = str(sample['question']) + str(sample['answer'])
          hash_value = hashlib.sha256(content.encode()).hexdigest()
          return f"qa:{hash_value}"
      # Fallback to full sample hash
      content = json.dumps(sample, sort_keys=True)
      hash_value = hashlib.sha256(content.encode()).hexdigest()
      return f"hash:{hash_value}"
  
  def _filter_remaining_samples(self, samples, existing_results):
      """Filter out samples that have already been processed."""
      if not existing_results:
          return samples, []
      
      # Build a dict of processed sample keys to results
      processed_results = {}
      for result in existing_results:
          # Only consider completed samples (those with responses)
          if "response" in result:
              key = self._get_sample_key(result)
              processed_results[key] = result
      
      # Filter samples and build ordered lists
      remaining_samples = []
      existing_samples_ordered = []
      for sample in samples:
          key = self._get_sample_key(sample)
          if key in processed_results:
              # Keep in original order
              existing_samples_ordered.append(processed_results[key])
          else:
              remaining_samples.append(sample)
      
      if len(remaining_samples) < len(samples):
          print(f"Resume: Skipping {len(samples) - len(remaining_samples)} already-processed samples")
          print(f"Resume: Processing {len(remaining_samples)} remaining samples")
      
      return remaining_samples, existing_samples_ordered

  def eval(self):
      model = self.model
      dataset_path = self.dataset_path
      output_path = self.output_path
      num_chunks = self.num_chunks
      chunk_idx = self.chunk_idx
      if num_chunks == 1:
          results_path = os.path.join(output_path,"results.json")
          matric_path = os.path.join(output_path,"metrics.json")
          
          # Check for existing results and resume if found
          existing_results = self._load_existing_results(results_path)
          remaining_samples, completed_samples = self._filter_remaining_samples(self.samples, existing_results)
          
          # Process only remaining samples
          new_out_samples = []
          if remaining_samples:
              new_out_samples = self.run(remaining_samples, model)
          else:
              print("All samples already processed. Skipping inference.")
          
          # Reconstruct results in original sample order
          # Build lookup dicts
          completed_by_key = {self._get_sample_key(s): s for s in completed_samples}
          new_by_key = {self._get_sample_key(s): s for s in new_out_samples}
          
          # Reconstruct in original order
          out_samples = []
          for sample in self.samples:
              key = self._get_sample_key(sample)
              if key in completed_by_key:
                  out_samples.append(completed_by_key[key])
              elif key in new_by_key:
                  out_samples.append(new_by_key[key])
              else:
                  # This shouldn't happen, but handle gracefully
                  print(f"Warning: Sample with key {key} not found in completed or new results")
          
          if len(out_samples) != len(self.samples):
              print(f"Warning: Result count mismatch. Expected {len(self.samples)}, got {len(out_samples)}")
          
          save_json(results_path,out_samples)

          metrics,out_samples = self.cal_metrics(out_samples)
          save_json(matric_path,metrics)
          save_json(results_path,out_samples)
          return metrics

      elif num_chunks > 1:
        results_path = os.path.join(output_path,f"results_{chunk_idx}.json")
        final_results_path = os.path.join(output_path,"results.json")
        
        # Check for existing chunk results and resume if found
        existing_results = self._load_existing_results(results_path)
        remaining_samples, completed_samples = self._filter_remaining_samples(self.samples, existing_results)
        
        # Process only remaining samples
        new_out_samples = []
        if remaining_samples:
            new_out_samples = self.run(remaining_samples, model)
        else:
            print(f"Chunk {chunk_idx}: All samples already processed. Skipping inference.")
        
        # Reconstruct results in original sample order
        # Build lookup dicts
        completed_by_key = {self._get_sample_key(s): s for s in completed_samples}
        new_by_key = {self._get_sample_key(s): s for s in new_out_samples}
        
        # Reconstruct in original order
        out_samples = []
        for sample in self.samples:
            key = self._get_sample_key(sample)
            if key in completed_by_key:
                out_samples.append(completed_by_key[key])
            elif key in new_by_key:
                out_samples.append(new_by_key[key])
            else:
                # This shouldn't happen, but handle gracefully
                print(f"Warning: Sample with key {key} not found in completed or new results")
        
        if len(out_samples) != len(self.samples):
            print(f"Warning: Result count mismatch. Expected {len(self.samples)}, got {len(out_samples)}")
        
        save_json(results_path,out_samples)

        total_results_path = os.listdir(output_path)
        total_results_path = [result for result in total_results_path if result.startswith("results_")]
        if len(total_results_path) == num_chunks:
            total_results = []
            for result in total_results_path:
                results_path = os.path.join(output_path,result)
                with open(results_path,"r") as f:
                    total_results.extend(json.load(f))

            save_json(final_results_path,total_results)
            metrics,out_samples = self.cal_metrics(total_results)
            matric_path = os.path.join(output_path,"metrics.json")
            save_json(matric_path,metrics)
            save_json(final_results_path,out_samples)
            return metrics
        else:
            return None
      else:
          raise ValueError("num_chunks must be greater than 0")
  
  def _download_file_local(self,local_path,url):
        # download the specific file to local_path
        
        os.makedirs(local_path,exist_ok=True)
        
        # Extract filename from URL
        filename = url.split("/")[-1]
        file_path = os.path.join(local_path, filename)
        
        # Check if wget or curl is available
        if os.system("which wget > /dev/null 2>&1") == 0:
            download_cmd = f"wget {url} -O {file_path}"
        elif os.system("which curl > /dev/null 2>&1") == 0:
            download_cmd = f"curl -L {url} -o {file_path}"
        else:
            raise RuntimeError("Neither wget nor curl is available for downloading")

        # Download with error handling
        if os.system(download_cmd) != 0:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise RuntimeError("Failed to download dataset")
        
  def _unzip_img_zip_local(self, local_path, zip_filename):
        # suppose zip_filename is like 'images.zip' or 'data.tgz'
        zip_file_path = os.path.join(local_path, zip_filename)
        
        if zip_filename.endswith('.zip'):
            if os.system(f"unzip -q {zip_file_path} -d {local_path}") != 0:
                if os.path.exists(zip_file_path):
                    os.remove(zip_file_path)
                raise RuntimeError("Failed to extract dataset")
        elif zip_filename.endswith('.tgz') or zip_filename.endswith('.tar.gz'):
            if os.system(f"tar -xzf {zip_file_path} -C {local_path}") != 0:
                if os.path.exists(zip_file_path):
                    os.remove(zip_file_path)
                raise RuntimeError("Failed to extract dataset")
        else:
            if os.path.exists(zip_file_path):
                os.remove(zip_file_path)
            raise RuntimeError(f"Unsupported file format: {zip_filename}")
        
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)