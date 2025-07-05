# Rainfall-runoff modeling on H.J. Andrews using EA-LSTM

## Downloading Streamflow Data (HF004)

This project uses **daily streamflow summaries** from the H.J. Andrews Experimental Forest, available through the Environmental Data Initiative (EDI). Due to repository restrictions, this dataset must be **downloaded manually**.

Please follow these steps:

1. Visit the dataset page on EDI:  
  
    [https://portal.edirepository.org/nis/mapbrowse?packageid=knb-lter-and.4341.35](https://portal.edirepository.org/nis/mapbrowse?packageid=knb-lter-and.4341.35)

2. Scroll to the list of **Data Entities** and locate:  
  
    **Daily streamflow summaries** (Entity 2, approximately 14.7 MiB)

3. Click the **Download** icon next to that entry.

4. Save the file as:  
  
    `HF004_daily_streamflow_entity2.csv`

5. Move the file to the following directory within this project:
  
    `data/streamflow/HF004_daily_streamflow_entity2.csv`

### Citation

> Johnson, S.L., S.M. Wondzell, and J.S. Rothacher. 2023. Stream discharge in gaged watersheds at the HJ Andrews Experimental Forest, 1949 to present ver 35. Environmental Data Initiative. <https://doi.org/10.6073/pasta/62bd85397e23bc9c50c7f652c0cc58ea> (Accessed 2025-07-05).
