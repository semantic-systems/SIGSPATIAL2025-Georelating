# LLM Agents for Georelating - A New Task for Locating Events
This repository contains the dataset and source code for the ACM SIGSPATIAL 2025 submission "LLM Agents for Georelating - A New Task for Locating Events" by Moltzen, Huang and Usbeck.

## GANDR: Georelating-annotated Natural Disaster Reports
GANDR is a silver-standard dataset of 2,000 synthetic disaster reports with annotated H3 DGGS cell indices and geospatial relations for the US and EU. 
Please find it in the `data/` directory. In case you use the dataset, please cite the Zenodo entryas:
```bibtex
Moltzen, K., Usbeck, R., & Huang, J. (2025). GANDR: Georelating-Annotated Natural Disaster Reports [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15612556
```

## Code usage
### Preliminaries
- Please ensure you have an envoronment file containing your API key for the LLM invocation `SAIA_API_KEY` and GeoNames API username `GEONAMES_USERNAME`.
- In case you want to run the code to geocode the LGL corpus or GeoCoDe dataset, please download and save it to `data/`.
- Note that the model roster of the ChatAI API changes over time: the models used for the published experiments (`llama-3.3-70b-instruct`, `mistral-large-instruct`) are no longer served. Choose currently available substitutes (see `GET /v1/models`, cached in `modules/available_models.json`) and set them as `actor`/`critic` in `modules/georelating.py`.
### Requirements
This code was written in Python 3.12. To run the code, please install the requirements from `requirements.txt`:
```bash
pip install -r requirements.txt
```
### Structure of the Code
The code is structured as follows:
- `agent_components/`: Contains the code for the internal and external environments, LLM agents, and hybrid memory used in the experiments.
- `data/`: Contains the GANDR corpus, as well as the few-shot examples used in the experiments.
- `data_handler/`: Contains the code for constructing the GANDR corpus, as well as for loading the LGL and GeoCodDe datasets.'
- `evaluation/`: Contains the code for evaluating the results of the experiments, split into geocoding and georelating.
- `helpers/`: Contains helper functions for the experiments.
- `models/`: Contains the code for the Pydantic models defined for validating the in- and outputs.
- `modules/`: Contains the code for the main modules used in the experiments.
- `output/`: The directory to store the output of the experiments.

### Running the Code
The main module is the `georelating.py` file, which contains the main function to run the Georelating experiments.
If you want to run candidate generation or resolution separately, you can use the `geocoding.py` files.

### Demo API with map UI
`demo/` contains a FastAPI app exposing the pipeline as an API plus a Leaflet map frontend
(select a GANDR article or paste your own report, optionally with structured LLM output):
```bash
uvicorn demo.server:app --host 127.0.0.1 --port 8123   # then open http://localhost:8123
```
See `demo/DEPLOYMENT.md` for running it on a (GPU) server.