"""Knowledge Graph — Entitäten, Relationen, Traversal

SQLite-basierter Knowledge Graph mit:
- Entitäten (Personen, Projekte, Tools, Konzepte, Orte, ...)
- Typisierte Relationen (nutzt, arbeitet_an, kennt, gehört_zu, ...)
- Memory-Entity Linking
- BFS Traversal + Shortest Path
- LLM-basierte Entity-Extraktion via Ollama
"""
import sqlite3
import json
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from collections import deque
import httpx


class KnowledgeGraph:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_tables()
    
    def _init_tables(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL DEFAULT 'concept',
                properties TEXT DEFAULT '{}',
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES entities(id),
                FOREIGN KEY (target_id) REFERENCES entities(id)
            );
            
            CREATE TABLE IF NOT EXISTS entity_mentions (
                entity_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                project TEXT DEFAULT 'default',
                created_at TEXT NOT NULL,
                PRIMARY KEY (entity_id, memory_id),
                FOREIGN KEY (entity_id) REFERENCES entities(id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
            CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id);
            CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id);
            CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);
            CREATE INDEX IF NOT EXISTS idx_mentions_entity ON entity_mentions(entity_id);
            CREATE INDEX IF NOT EXISTS idx_mentions_memory ON entity_mentions(memory_id);
        """)
        conn.commit()
        conn.close()
    
    def _make_id(self, *parts: str) -> str:
        raw = ":".join(str(p).lower().strip() for p in parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
    
    # ──────────────────────────────────────────────────────────
    # Entity CRUD
    # ──────────────────────────────────────────────────────────
    
    def add_entity(self, name: str, entity_type: str = "concept",
                   properties: Optional[dict] = None) -> dict:
        eid = self._make_id(entity_type, name)
        now = datetime.now().isoformat()
        props = json.dumps(properties or {})
        
        conn = sqlite3.connect(self.db_path)
        existing = conn.execute("SELECT id FROM entities WHERE id = ?", (eid,)).fetchone()
        if existing:
            conn.execute(
                "UPDATE entities SET properties = ?, last_seen = ? WHERE id = ?",
                (props, now, eid)
            )
        else:
            conn.execute(
                "INSERT INTO entities (id, name, type, properties, first_seen, last_seen) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (eid, name, entity_type, props, now, now)
            )
        conn.commit()
        conn.close()
        return {"id": eid, "name": name, "type": entity_type}
    
    def get_entity(self, name: str = None, entity_id: str = None) -> Optional[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        if entity_id:
            row = conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
        elif name:
            row = conn.execute(
                "SELECT * FROM entities WHERE LOWER(name) = LOWER(?)", (name,)
            ).fetchone()
        else:
            conn.close()
            return None
        conn.close()
        if row:
            d = dict(row)
            d["properties"] = json.loads(d.get("properties", "{}"))
            return d
        return None
    
    def search_entities(self, query: str, entity_type: str = None,
                        limit: int = 25) -> list:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        sql = "SELECT * FROM entities WHERE LOWER(name) LIKE ?"
        params = [f"%{query.lower()}%"]
        if entity_type:
            sql += " AND type = ?"
            params.append(entity_type)
        sql += " ORDER BY last_seen DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    
    def list_entities(self, entity_type: str = None, limit: int = 100) -> list:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        if entity_type:
            rows = conn.execute(
                "SELECT * FROM entities WHERE type = ? ORDER BY last_seen DESC LIMIT ?",
                (entity_type, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM entities ORDER BY last_seen DESC LIMIT ?",
                (limit,)
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    
    def delete_entity(self, name: str = None, entity_id: str = None) -> bool:
        ent = self.get_entity(name=name, entity_id=entity_id)
        if not ent:
            return False
        eid = ent["id"]
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM relations WHERE source_id = ? OR target_id = ?", (eid, eid))
        conn.execute("DELETE FROM entity_mentions WHERE entity_id = ?", (eid,))
        conn.execute("DELETE FROM entities WHERE id = ?", (eid,))
        conn.commit()
        conn.close()
        return True
    
    # ──────────────────────────────────────────────────────────
    # Relations
    # ──────────────────────────────────────────────────────────
    
    def add_relation(self, source_name: str, target_name: str,
                     relation_type: str,
                     source_type: str = "concept", target_type: str = "concept",
                     properties: Optional[dict] = None) -> dict:
        src = self.add_entity(source_name, source_type)
        tgt = self.add_entity(target_name, target_type)
        rid = self._make_id(src["id"], relation_type, tgt["id"])
        now = datetime.now().isoformat()
        props = json.dumps(properties or {})
        
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO relations "
            "(id, source_id, target_id, relation_type, properties, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (rid, src["id"], tgt["id"], relation_type, props, now)
        )
        conn.commit()
        conn.close()
        return {
            "id": rid,
            "source": src, "target": tgt,
            "relation_type": relation_type
        }
    
    def get_relations(self, entity_name: str = None, entity_id: str = None,
                      relation_type: str = None,
                      direction: str = "both") -> list:
        """Hole alle Relationen einer Entität.
        direction: 'out' (ausgehend), 'in' (eingehend), 'both'
        """
        ent = self.get_entity(name=entity_name, entity_id=entity_id)
        if not ent:
            return []
        eid = ent["id"]
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        results = []
        
        if direction in ("out", "both"):
            sql = (
                "SELECT r.*, e.name AS target_name, e.type AS target_type "
                "FROM relations r JOIN entities e ON r.target_id = e.id "
                "WHERE r.source_id = ?"
            )
            params = [eid]
            if relation_type:
                sql += " AND r.relation_type = ?"
                params.append(relation_type)
            for row in conn.execute(sql, params):
                d = dict(row)
                d["direction"] = "out"
                d["other_name"] = d["target_name"]
                d["other_type"] = d["target_type"]
                results.append(d)
        
        if direction in ("in", "both"):
            sql = (
                "SELECT r.*, e.name AS source_name, e.type AS source_type "
                "FROM relations r JOIN entities e ON r.source_id = e.id "
                "WHERE r.target_id = ?"
            )
            params = [eid]
            if relation_type:
                sql += " AND r.relation_type = ?"
                params.append(relation_type)
            for row in conn.execute(sql, params):
                d = dict(row)
                d["direction"] = "in"
                d["other_name"] = d["source_name"]
                d["other_type"] = d["source_type"]
                results.append(d)
        
        conn.close()
        return results
    
    # ──────────────────────────────────────────────────────────
    # Memory ↔ Entity Linking
    # ──────────────────────────────────────────────────────────
    
    def link_memory(self, memory_id: str, entity_name: str,
                    entity_type: str = "concept",
                    project: str = "default") -> dict:
        ent = self.add_entity(entity_name, entity_type)
        now = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR IGNORE INTO entity_mentions "
            "(entity_id, memory_id, project, created_at) VALUES (?, ?, ?, ?)",
            (ent["id"], memory_id, project, now)
        )
        conn.commit()
        conn.close()
        return {"entity": ent, "memory_id": memory_id}
    
    def get_entity_memories(self, entity_name: str = None,
                            entity_id: str = None) -> list:
        ent = self.get_entity(name=entity_name, entity_id=entity_id)
        if not ent:
            return []
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT memory_id, project FROM entity_mentions WHERE entity_id = ?",
            (ent["id"],)
        ).fetchall()
        conn.close()
        return [{"memory_id": r[0], "project": r[1]} for r in rows]
    
    def get_memory_entities(self, memory_id: str) -> list:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT e.* FROM entities e "
            "JOIN entity_mentions em ON e.id = em.entity_id "
            "WHERE em.memory_id = ?",
            (memory_id,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    
    # ──────────────────────────────────────────────────────────
    # Graph Traversal
    # ──────────────────────────────────────────────────────────
    
    def find_connected(self, entity_name: str, max_depth: int = 2,
                       relation_type: str = None) -> dict:
        """BFS: Finde alle verbundenen Entitäten bis max_depth Hops."""
        start = self.get_entity(name=entity_name)
        if not start:
            return {"error": f"Entity '{entity_name}' nicht gefunden"}
        
        visited = {}
        queue = deque()
        edges = []
        
        visited[start["id"]] = {"entity": start, "depth": 0}
        queue.append((start["id"], 0))
        
        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue
            
            rels = self.get_relations(entity_id=current_id, relation_type=relation_type)
            for rel in rels:
                other_id = (
                    rel["target_id"] if rel["direction"] == "out"
                    else rel["source_id"]
                )
                edges.append({
                    "from": current_id,
                    "to": other_id,
                    "type": rel["relation_type"],
                    "direction": rel["direction"]
                })
                
                if other_id not in visited:
                    other_ent = self.get_entity(entity_id=other_id)
                    if other_ent:
                        visited[other_id] = {
                            "entity": other_ent,
                            "depth": depth + 1
                        }
                        queue.append((other_id, depth + 1))
        
        return {
            "root": start["name"],
            "nodes": list(visited.values()),
            "edges": edges,
            "total_nodes": len(visited),
            "total_edges": len(edges)
        }
    
    def find_path(self, from_name: str, to_name: str,
                  max_depth: int = 6) -> dict:
        """BFS Shortest Path zwischen zwei Entitäten."""
        start = self.get_entity(name=from_name)
        end = self.get_entity(name=to_name)
        if not start:
            return {"error": f"Start-Entity '{from_name}' nicht gefunden"}
        if not end:
            return {"error": f"Ziel-Entity '{to_name}' nicht gefunden"}
        if start["id"] == end["id"]:
            return {"path": [start], "edges": [], "length": 0}
        
        # BFS mit Rückverfolgung
        visited = {start["id"]: None}
        queue = deque([(start["id"], 0)])
        
        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue
            
            rels = self.get_relations(entity_id=current_id)
            for rel in rels:
                other_id = (
                    rel["target_id"] if rel["direction"] == "out"
                    else rel["source_id"]
                )
                if other_id not in visited:
                    visited[other_id] = {
                        "from": current_id,
                        "relation": rel["relation_type"],
                        "direction": rel["direction"]
                    }
                    if other_id == end["id"]:
                        return self._reconstruct_path(visited, start, end)
                    queue.append((other_id, depth + 1))
        
        return {"path": [], "edges": [], "length": -1,
                "message": f"Kein Pfad zwischen '{from_name}' und '{to_name}' (max_depth={max_depth})"}
    
    def _reconstruct_path(self, visited: dict, start: dict, end: dict) -> dict:
        path = []
        edge_list = []
        current = end["id"]
        
        while current != start["id"]:
            ent = self.get_entity(entity_id=current)
            step = visited[current]
            path.append(ent)
            edge_list.append({
                "from": step["from"],
                "to": current,
                "relation": step["relation"],
                "direction": step["direction"]
            })
            current = step["from"]
        path.append(start)
        path.reverse()
        edge_list.reverse()
        
        return {"path": path, "edges": edge_list, "length": len(path) - 1}
    
    def get_subgraph(self, entity_names: list, max_depth: int = 1) -> dict:
        """Subgraph um mehrere Entitäten."""
        all_nodes = {}
        all_edges = []
        
        for name in entity_names:
            result = self.find_connected(name, max_depth=max_depth)
            if "error" not in result:
                for node in result["nodes"]:
                    nid = node["entity"]["id"]
                    if nid not in all_nodes:
                        all_nodes[nid] = node
                all_edges.extend(result["edges"])
        
        # Dedupliziere Edges
        seen = set()
        unique_edges = []
        for e in all_edges:
            key = (e["from"], e["to"], e["type"])
            if key not in seen:
                seen.add(key)
                unique_edges.append(e)
        
        return {
            "nodes": list(all_nodes.values()),
            "edges": unique_edges,
            "total_nodes": len(all_nodes),
            "total_edges": len(unique_edges)
        }
    
    # ──────────────────────────────────────────────────────────
    # LLM Entity Extraction
    # ──────────────────────────────────────────────────────────
    
    def extract_and_link(self, memory_id: str, content: str,
                         project: str = "default",
                         ollama_url: str = "http://localhost:11434",
                         model: str = "qwen2.5:32b") -> dict:
        """Extrahiere Entitäten + Relationen aus Memory-Text via Ollama LLM."""
        prompt = f"""Analysiere diesen Text und extrahiere Entitäten und Relationen.

TEXT:
{content}

Antworte NUR mit einem JSON-Objekt in diesem Format (keine Erklärungen):
{{
  "entities": [
    {{"name": "Beispiel", "type": "person|project|tool|company|concept|place|technology"}}
  ],
  "relations": [
    {{"source": "Entity1", "target": "Entity2", "type": "nutzt|arbeitet_an|kennt|gehört_zu|basiert_auf|erstellt|befindet_in"}}
  ]
}}

Regeln:
- Nur konkrete, benannte Entitäten (keine generischen Begriffe)
- Typen: person, project, tool, company, concept, place, technology
- Relations-Typen: nutzt, arbeitet_an, kennt, gehört_zu, basiert_auf, erstellt, befindet_in, hat, ist
- Maximal 10 Entitäten und 10 Relationen
- Antworte NUR mit dem JSON, kein Markdown, kein Text drumherum
"""
        
        try:
            r = httpx.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 1024}
                },
                timeout=60.0
            )
            if r.status_code != 200:
                return {"entities": 0, "relations": 0, "error": f"HTTP {r.status_code}"}
            
            response_text = r.json().get("response", "")
            data = self._parse_extraction(response_text)
            
            if not data:
                return {"entities": 0, "relations": 0, "error": "Parse-Fehler"}
            
            # Entitäten speichern + linken
            entity_count = 0
            for ent in data.get("entities", []):
                name = ent.get("name", "").strip()
                etype = ent.get("type", "concept").strip()
                if name and len(name) > 1:
                    self.add_entity(name, etype)
                    self.link_memory(memory_id, name, etype, project)
                    entity_count += 1
            
            # Relationen speichern
            relation_count = 0
            for rel in data.get("relations", []):
                src = rel.get("source", "").strip()
                tgt = rel.get("target", "").strip()
                rtype = rel.get("type", "").strip()
                if src and tgt and rtype:
                    self.add_relation(src, tgt, rtype)
                    relation_count += 1
            
            return {"entities": entity_count, "relations": relation_count}
        
        except httpx.ConnectError:
            return {"entities": 0, "relations": 0, "error": "Ollama nicht erreichbar"}
        except Exception as e:
            return {"entities": 0, "relations": 0, "error": str(e)}
    
    def _parse_extraction(self, text: str) -> Optional[dict]:
        """Parse JSON aus LLM-Antwort (robust gegen Markdown-Wrapper)."""
        # Versuche direktes JSON
        text = text.strip()
        
        # Entferne Markdown Code-Block wenn vorhanden
        if text.startswith("```"):
            lines = text.split("\n")
            # Entferne erste und letzte Zeile
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Finde JSON-Block in der Antwort
        match = re.search(r'\{[^{}]*"entities"[^{}]*\[.*?\].*?\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        return None
    
    # ──────────────────────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────────────────────
    
    def stats(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        relations = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        mentions = conn.execute("SELECT COUNT(*) FROM entity_mentions").fetchone()[0]
        
        entity_types = {}
        for row in conn.execute(
            "SELECT type, COUNT(*) FROM entities GROUP BY type ORDER BY COUNT(*) DESC"
        ):
            entity_types[row[0]] = row[1]
        
        relation_types = {}
        for row in conn.execute(
            "SELECT relation_type, COUNT(*) FROM relations GROUP BY relation_type ORDER BY COUNT(*) DESC"
        ):
            relation_types[row[0]] = row[1]
        
        conn.close()
        return {
            "entities": entities,
            "relations": relations,
            "memory_links": mentions,
            "entity_types": entity_types,
            "relation_types": relation_types
        }
