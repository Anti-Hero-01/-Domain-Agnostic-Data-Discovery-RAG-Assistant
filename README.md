# Setting Up Credentials

## OpenAI API Key

1. Visit https://platform.openai.com/account/api-keys
2. Sign up or log in to your OpenAI account
3. Click "Create new secret key"
4. Copy the generated key (you won't be able to see it again)
5. In `.env` file, replace `your_key_here` with your actual OpenAI API key

## Neo4j Database Setup

1. Download and install Neo4j Desktop from https://neo4j.com/download/
2. Create a new project in Neo4j Desktop
3. Add a new database to your project
4. Set a secure password when creating the database
5. In `.env` file, replace `your_neo4j_password` with your database password

## Environment Variables

Copy the `.env.example` file to `.env` and fill in your credentials:

```
OPENAI_API_KEY=sk-...your-key-here...
NEO4J_PASSWORD=your-database-password
```

⚠️ Security Notes:

- Never commit your `.env` file to version control
- Keep your API keys secure and don't share them
- Rotate your keys periodically for better security
