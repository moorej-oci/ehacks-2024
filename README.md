# eHacks-2024

This repository is a showcase of [LangChain](https://www.langchain.com/langchain) for eHacks 2024 accompanying a presentation.

## Developer Setup

This project uses [Poetry](https://python-poetry.org/) to manage dependencies. Follow the installation instructions on their website or utilize the provided [requirements.txt](./requirements.txt) to install in an existing environment with `pip install -r requirements.txt`.

Make a copy of [.env.example](./env.example) with `cp .env.example .env` (macos/linux) or copy in your IDE/file browser of choice. Then populate the `.env` file with your OpenAI API Key (Instructions for setting up an account: https://platform.openai.com/docs/quickstart/account-setup) 

## Generating Requirements

Run the following command to create a requirements file with poetry for those that may not wish to use poetry with this project.
`poetry export -f requirements.txt --output requirements.txt --without-hashes`
