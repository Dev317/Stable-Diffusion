# Audio Diffuser

An audio-thumbnail generator.

Currently, the application is able to classify these 10 sounds:

- Air Conditioner
- Car Horn
- Children Playing
- Dog bark
- Drilling
- Engine Idling
- Gun Shot
- Jackhammer
- Siren
- Street Music

## Server

### Requirements

A machine that is capable of running TensorFlow and PyTorch.

### Installation

Navigate to `/server`.

Install all dependencies within `requirements.txt`. This step depends on the system architecture you are currently using.

If using Anaconda / Minicoda, create a new environment specifically for this project:

```bash
$ conda env create -f environment.yml
```

Activate the `audio-diffusion` environment.

```bash
$ conda activate audio-diffusion
```

Download the pre-trained models.

```bash
$ python3 download.py
```

Lastly, set the environment variables required by copying `.env.sample` into `.env`.

### Run

```bash
$ python3 app.py
```

The server will be accessible at `http://localhost:5001`.

## Web app

### Requirements

- `npm` or `yarn`
- Node

### Installation

Navigate to `/webapp`.

```bash
$ yarn install
```

Configure `.env.local` to set whether the application should use a server mock.

### Run

```bash
$ yarn dev
```

The web application will be accessible at `http://localhost:3000`.
