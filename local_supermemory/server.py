"""Local Supermemory MCP Server v2 — mit Knowledge Graph + Ollama Embeddings"""
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .store import MemoryStore
from .profile import get_engine

server = Server("local-supermemory")
store = MemoryStore()


# ═══════════════════════════════════════════════════════════════
# Tool Definitions
# ═══════════════════════════════════════════════════════════════

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        # ── Memory Tools (v1) ────────────────────────────────
        Tool(
            name="memory",
            description="Speichere oder vergesse Informationen. action: 'save'|'forget', content: Text, project: optional",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["save", "forget"], "default": "save"},
                    "content": {"type": "string", "maxLength": 200000},
                    "project": {"type": "string", "maxLength": 128}
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="recall",
            description="Semantische Suche in Memories mit Profil-Aggregation",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "maxLength": 1000},
                    "project": {"type": "string", "maxLength": 128},
                    "include_profile": {"type": "boolean", "default": True},
                    "n_results": {"type": "integer", "default": 15}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="list_projects",
            description="Liste alle Projekte",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="stats",
            description="Speicher-Statistiken",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="whoami",
            description="Zeigt Benutzerinfo basierend auf Memories",
            inputSchema={"type": "object", "properties": {}}
        ),

        # ── Knowledge Graph Tools (v2) ───────────────────────
        Tool(
            name="graph_add_entity",
            description="Erstelle oder aktualisiere eine Entität im Knowledge Graph. name: Name, type: z.B. person/project/tool/company/concept, properties: optionales dict",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "default": "unknown"},
                    "properties": {"type": "object", "default": {}}
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="graph_add_relation",
            description="Erstelle eine Relation zwischen zwei Entitäten. source/target: Namen, relation_type: z.B. 'arbeitet_an', 'nutzt', 'kennt', 'gehört_zu'",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relation_type": {"type": "string"},
                    "source_type": {"type": "string", "default": "unknown"},
                    "target_type": {"type": "string", "default": "unknown"},
                    "properties": {"type": "object", "default": {}}
                },
                "required": ["source", "target", "relation_type"]
            }
        ),
        Tool(
            name="graph_link_memory",
            description="Verknüpfe eine Memory (per ID) mit einer Entität",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "entity_name": {"type": "string"},
                    "entity_type": {"type": "string", "default": "unknown"},
                    "project": {"type": "string", "default": "default"}
                },
                "required": ["memory_id", "entity_name"]
            }
        ),
        Tool(
            name="graph_query",
            description="Abfrage des Knowledge Graphs. action: 'find_connected'|'shortest_path'|'subgraph'|'relations'|'search'|'entity_memories'",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["find_connected", "shortest_path", "subgraph",
                                 "relations", "search", "entity_memories"]
                    },
                    "entity": {"type": "string", "description": "Entitäts-Name für find_connected, relations, entity_memories"},
                    "from_entity": {"type": "string", "description": "Start für shortest_path"},
                    "to_entity": {"type": "string", "description": "Ziel für shortest_path"},
                    "entities": {"type": "array", "items": {"type": "string"}, "description": "Liste für subgraph"},
                    "query": {"type": "string", "description": "Suchbegriff für search"},
                    "entity_type": {"type": "string", "description": "Typ-Filter"},
                    "relation_type": {"type": "string", "description": "Relations-Typ-Filter"},
                    "max_depth": {"type": "integer", "default": 2}
                },
                "required": ["action"]
            }
        ),
        Tool(
            name="graph_stats",
            description="Knowledge Graph Statistiken: Entitäten, Relationen, Typen",
            inputSchema={"type": "object", "properties": {}}
        ),
    ]


# ═══════════════════════════════════════════════════════════════
# Tool Handlers
# ═══════════════════════════════════════════════════════════════

@server.call_tool()
async def call_tool(name: str, args: dict) -> list[TextContent]:

    # ── Memory Tools ─────────────────────────────────────────
    if name == "memory":
        action = args.get("action", "save")
        content = args.get("content", "")
        project = args.get("project", "default")
        if not content:
            return [TextContent(type="text", text="Fehler: content erforderlich")]
        if action == "save":
            r = store.save(content, project)
            return [TextContent(type="text", text=f"✅ Gespeichert (ID: {r['id']}, Projekt: {r['project']})")]
        else:
            r = store.forget(content, project)
            return [TextContent(type="text", text=f"{'✅ Gelöscht' if r['deleted'] else '⚠️ Nicht gefunden'} (ID: {r['id']})")]

    elif name == "recall":
        query = args.get("query", "")
        project = args.get("project", "default")
        incl_profile = args.get("include_profile", True)
        n = args.get("n_results", 15)

        if not query:
            return [TextContent(type="text", text="Fehler: query erforderlich")]

        mems = store.recall(query, project, n)
        parts = []

        if incl_profile:
            cached = store.get_profile(project)
            if cached:
                profile = cached
            else:
                all_mems = store.get_all(project)
                if all_mems:
                    profile = await get_engine().generate(all_mems)
                    store.set_profile(project, profile)
                else:
                    profile = "Keine Memories."
            parts.append("## User Profile\n" + profile + "\n")

        parts.append("## Relevante Memories\n")
        if mems:
            for i, m in enumerate(mems, 1):
                parts.append(f"### {i}. ({m['similarity']}%)\n{m['content']}\n")
        else:
            parts.append("Keine gefunden.\n")

        return [TextContent(type="text", text="\n".join(parts))]

    elif name == "list_projects":
        projects = store.list_projects()
        lines = ["## Projekte\n"]
        for p in projects:
            lines.append(f"- **{p['name']}**: {p['count']} Memories")
        return [TextContent(type="text", text="\n".join(lines))]

    elif name == "stats":
        s = store.stats()
        text = f"""## Statistiken
**Gesamt:** {s['total']} Memories in {s['projects']} Projekten
**Speicher:** `{s['path']}`
**Embeddings:** {s['embedding_provider']}
**Knowledge Graph:** {s['graph']['entities']} Entitäten, {s['graph']['relations']} Relationen, {s['graph']['memory_links']} Memory-Links
"""
        if s['graph']['entity_types']:
            text += f"**Entitäts-Typen:** {', '.join(s['graph']['entity_types'])}\n"
        if s['graph']['relation_types']:
            text += f"**Relations-Typen:** {', '.join(s['graph']['relation_types'])}\n"
        text += "\n"
        for proj, cnt in s['by_project'].items():
            text += f"- {proj}: {cnt}\n"
        return [TextContent(type="text", text=text)]

    elif name == "whoami":
        mems = store.get_all("default")[:10]
        if mems:
            profile = await get_engine().generate(mems)
            return [TextContent(type="text", text=f"## Benutzer-Info\n{profile}")]
        return [TextContent(type="text", text="Keine Memories vorhanden.")]

    # ── Knowledge Graph Tools ────────────────────────────────
    elif name == "graph_add_entity":
        r = store.graph.add_entity(
            name=args["name"],
            entity_type=args.get("type", "unknown"),
            properties=args.get("properties", {})
        )
        return [TextContent(type="text",
            text=f"✅ Entität: **{r['name']}** (Typ: {r['type']}, ID: {r['id']})")]

    elif name == "graph_add_relation":
        r = store.graph.add_relation(
            source_name=args["source"],
            target_name=args["target"],
            relation_type=args["relation_type"],
            source_type=args.get("source_type", "unknown"),
            target_type=args.get("target_type", "unknown"),
            properties=args.get("properties", {})
        )
        return [TextContent(type="text",
            text=f"✅ Relation: **{r['source']['name']}** —[{r['relation_type']}]→ **{r['target']['name']}**")]

    elif name == "graph_link_memory":
        r = store.graph.link_memory(
            memory_id=args["memory_id"],
            entity_name=args["entity_name"],
            entity_type=args.get("entity_type", "unknown"),
            project=args.get("project", "default")
        )
        return [TextContent(type="text",
            text=f"✅ Memory {r['memory_id']} ↔ Entität **{r['entity']['name']}**")]

    elif name == "graph_query":
        return [TextContent(type="text", text=_handle_graph_query(args))]

    elif name == "graph_stats":
        gs = store.graph.graph_stats()
        text = f"""## Knowledge Graph
**Entitäten:** {gs['entities']}
**Relationen:** {gs['relations']}
**Memory-Links:** {gs['memory_links']}
**Entitäts-Typen:** {', '.join(gs['entity_types']) if gs['entity_types'] else '—'}
**Relations-Typen:** {', '.join(gs['relation_types']) if gs['relation_types'] else '—'}"""
        return [TextContent(type="text", text=text)]

    return [TextContent(type="text", text=f"Unbekannt: {name}")]


# ═══════════════════════════════════════════════════════════════
# Graph Query Handler
# ═══════════════════════════════════════════════════════════════

def _handle_graph_query(args: dict) -> str:
    action = args["action"]
    g = store.graph

    if action == "find_connected":
        entity = args.get("entity", "")
        if not entity:
            return "Fehler: 'entity' erforderlich"
        r = g.find_connected(
            entity, max_depth=args.get("max_depth", 2),
            relation_type=args.get("relation_type")
        )
        if "error" in r:
            return f"⚠️ {r['error']}"
        lines = [f"## Verbunden mit '{r['root']}' ({r['total_nodes']} Knoten, {r['total_edges']} Kanten)\n"]
        for node in r["nodes"]:
            e = node["entity"]
            depth_marker = "  " * node["depth"] + ("→ " if node["depth"] > 0 else "● ")
            lines.append(f"{depth_marker}**{e['name']}** ({e.get('type', '?')}) [Tiefe {node['depth']}]")
        return "\n".join(lines)

    elif action == "shortest_path":
        fr = args.get("from_entity", "")
        to = args.get("to_entity", "")
        if not fr or not to:
            return "Fehler: 'from_entity' und 'to_entity' erforderlich"
        r = g.find_shortest_path(fr, to, max_depth=args.get("max_depth", 5))
        if r.get("length", -1) < 0:
            return f"⚠️ Kein Pfad zwischen '{fr}' und '{to}'"
        lines = [f"## Pfad: {fr} → {to} (Länge: {r['length']})\n"]
        for i, step in enumerate(r["path"]):
            e = step["entity"]
            rel = step.get("via_relation")
            if rel:
                lines.append(f"  —[{rel}]→ **{e['name']}** ({e.get('type', '?')})")
            else:
                lines.append(f"● **{e['name']}** ({e.get('type', '?')})")
        return "\n".join(lines)

    elif action == "subgraph":
        entities = args.get("entities", [])
        if not entities:
            return "Fehler: 'entities' Liste erforderlich"
        r = g.get_subgraph(entities, max_depth=args.get("max_depth", 1))
        lines = [f"## Subgraph ({r['total_nodes']} Knoten, {r['total_edges']} Kanten)\n"]
        lines.append("### Knoten")
        for node in r["nodes"]:
            e = node["entity"]
            lines.append(f"- **{e['name']}** ({e.get('type', '?')})")
        lines.append("\n### Kanten")
        for edge in r["edges"]:
            lines.append(f"- {edge['from'][:8]}… —[{edge['type']}]→ {edge['to'][:8]}…")
        return "\n".join(lines)

    elif action == "relations":
        entity = args.get("entity", "")
        if not entity:
            return "Fehler: 'entity' erforderlich"
        rels = g.get_relations(
            entity_name=entity,
            relation_type=args.get("relation_type")
        )
        if not rels:
            return f"Keine Relationen für '{entity}'"
        lines = [f"## Relationen von '{entity}' ({len(rels)})\n"]
        for rel in rels:
            if rel["direction"] == "out":
                lines.append(f"  → [{rel['relation_type']}] → **{rel.get('target_name', '?')}**")
            else:
                lines.append(f"  ← [{rel['relation_type']}] ← **{rel.get('source_name', '?')}**")
        return "\n".join(lines)

    elif action == "search":
        query = args.get("query", "")
        if not query:
            return "Fehler: 'query' erforderlich"
        results = g.search_entities(query, entity_type=args.get("entity_type"))
        if not results:
            return f"Keine Entitäten für '{query}'"
        lines = [f"## Entitäten-Suche: '{query}' ({len(results)} Treffer)\n"]
        for e in results:
            lines.append(f"- **{e['name']}** (Typ: {e['type']}, ID: {e['id']})")
        return "\n".join(lines)

    elif action == "entity_memories":
        entity = args.get("entity", "")
        if not entity:
            return "Fehler: 'entity' erforderlich"
        mems = g.get_entity_memories(entity_name=entity)
        if not mems:
            return f"Keine Memories für Entität '{entity}'"
        lines = [f"## Memories verknüpft mit '{entity}' ({len(mems)})\n"]
        for m in mems:
            lines.append(f"- Memory `{m['memory_id']}` (Projekt: {m['project']})")
        return "\n".join(lines)

    return f"Unbekannte Graph-Action: {action}"


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    asyncio.run(run())

async def run():
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())

if __name__ == "__main__":
    main()
