from neo4j import GraphDatabase
from typing import Dict, List
from core.config import Settings


class KnowledgeGraph:
    def __init__(self):
        settings = Settings()
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )

    def add_entities(self, entities: Dict[str, List[str]]):
        """Add entity nodes into the graph."""
        with self.driver.session() as session:
            for entity_type, values in entities.items():
                for value in values:
                    session.execute_write(self._create_entity, entity_type, value)

    def query_subgraph(self, query: str) -> List[Dict]:
        """Query the knowledge graph for relevant entities and relationships."""
        with self.driver.session() as session:
            return session.execute_read(self._query_graph, query)

    @staticmethod
    def _create_entity(tx, entity_type: str, value: str):
        cypher_query = (
            "MERGE (e:Entity {type: $type, value: $value}) "
            "RETURN e"
        )
        return tx.run(cypher_query, {"type": entity_type, "value": value})

    @staticmethod
    def _query_graph(tx, query: str) -> List[Dict]:
        """Simple example: find entities whose value contains the search query."""
        cypher_query = (
            "MATCH (e:Entity) "
            "WHERE e.value CONTAINS $query "
            "RETURN e.type AS type, e.value AS value"
        )
        result = tx.run(cypher_query, {"query": query})  # ✅ fixed
        return [dict(record) for record in result]

    def close(self):
        self.driver.close()
