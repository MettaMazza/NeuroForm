"""
BrainOrchestrator — Central Nervous System
=============================================

Coordinates the Five-Tier Memory System + Phase 2 systems + 9 brain systems
in a neuroscience-accurate processing pipeline.

Five-Tier Memory Architecture (ported from ErnOS V3/V4):
  T1: ContextStream — 500-turn persistent conversation history
  T2: VectorStore — Ollama embeddings + semantic search
  T3: KnowledgeGraph — Neo4j graph with entity relationships
  T4: LessonManager — Structured fact persistence
  T5: TapeMachine — 3D cognitive computation tape

Phase 2 Systems:
  CrossTierReconciler — LLM-powered conflict detection (T4 > T3 > T2)
  ScopeManager — Privacy scope enforcement (CORE_PRIVATE → PUBLIC)

Foundation Knowledge Injection:
  Extracts entities from user message, queries KG, injects as ground truth.

Processing pipeline per message:
  1. Circadian → NT modulation
  2. NT → LLM temperature, attention budget
  3. Foundation Knowledge → query KG for entities in message
  4. Salience → filter graph context
  5. Recall → gather context from all 4 tiers
  6. Habit Cache → check for cached response
  7. Predictive Model → predict user intent
  8. OllamaClient → LLM inference (with 4-tier context)
  9. Observe → persist turn to all tiers
  10. Neuroplasticity → apply feedback + baseline decay
"""
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from neuroform.memory.graph import KnowledgeGraph, GraphLayer
from neuroform.memory.context_stream import ContextStream
from neuroform.memory.vector_store import VectorStore
from neuroform.memory.lessons import LessonManager
from neuroform.memory.foundation import build_foundation_context
from neuroform.memory.tape_machine import TapeMachine
from neuroform.memory.reconciler import CrossTierReconciler
from neuroform.memory.scopes import ScopeManager, Scope
from neuroform.memory.amygdala import Amygdala
from neuroform.memory.salience_filter import SalienceScorer
from neuroform.memory.habit_cache import HabitCache
from neuroform.memory.neurotransmitters import NeurotransmitterState
from neuroform.memory.predictive_model import PredictiveModel
from neuroform.memory.default_mode_network import DefaultModeNetwork
from neuroform.memory.circadian import CircadianProfile
from neuroform.memory.neuroplasticity import AutonomousNeuroplasticity
from neuroform.llm.ollama_client import OllamaClient
import ollama

# ─── Tools & Effectors ─────────────────────────────────────────
from neuroform.tools.manager import tool_registry
import neuroform.tools.filesystem
import neuroform.tools.web
import neuroform.tools.terminal
import neuroform.tools.apple_script

logger = logging.getLogger(__name__)


# ─── Tool Call Interceptor ────────────────────────────────────────
import re as _re
_TOOL_PATTERN = _re.compile(r"\[TOOL:\s*\w+\(.*?\)\s*\]", _re.DOTALL)


def sanitize_tool_calls(text: str) -> str:
    """
    Strip ALL internal metadata from LLM output before serving to users.

    This is a critical safety net. The LLM sometimes echoes back internal
    context that was injected into the system prompt. None of this should
    ever reach the user. This function strips:

    - [TOOL: func(args)] patterns
    - [RECONCILIATION: ...] blocks
    - --- TAPE MACHINE ... --- blocks
    - [KNOWLEDGE GRAPH ...] / [/KNOWLEDGE GRAPH] blocks
    - [LESSONS ...] blocks
    - [CONVERSATION HISTORY] / [/CONVERSATION HISTORY] blocks
    - [OBSERVER-CRITIC ...] markers
    - [SYSTEM_INSTRUCTION]: ... lines

    Returns:
        The text with all internal metadata removed and cleaned up.
    """
    if not text:
        return text

    # Strip [TOOL: ...] calls
    cleaned = _TOOL_PATTERN.sub("", text)

    # Strip multi-line internal blocks using regex
    # [RECONCILIATION: ...] blocks (until next blank line or section)
    cleaned = _re.sub(
        r'\[RECONCILIATION:.*?\](?:\n(?:  [^\n]*\n)*)?',
        '', cleaned, flags=_re.DOTALL
    )

    # --- TAPE MACHINE ... --- blocks (until --- Total: or end of block)
    cleaned = _re.sub(
        r'--- TAPE MACHINE.*?--- Total:.*?---',
        '', cleaned, flags=_re.DOTALL
    )
    # Catch partial tape blocks
    cleaned = _re.sub(
        r'--- TAPE MACHINE.*?(?=\n\n|\Z)',
        '', cleaned, flags=_re.DOTALL
    )

    # [KNOWLEDGE GRAPH ...] to [/KNOWLEDGE GRAPH]
    cleaned = _re.sub(
        r'\[KNOWLEDGE GRAPH.*?\[/KNOWLEDGE GRAPH\]',
        '', cleaned, flags=_re.DOTALL
    )

    # [LESSONS ...] blocks
    cleaned = _re.sub(
        r'\[LESSONS.*?\](?:\n(?:  [^\n]*\n)*)?',
        '', cleaned, flags=_re.DOTALL
    )

    # [CONVERSATION HISTORY] to [/CONVERSATION HISTORY]
    cleaned = _re.sub(
        r'\[CONVERSATION HISTORY\].*?\[/CONVERSATION HISTORY\]',
        '', cleaned, flags=_re.DOTALL
    )

    # [OBSERVER-CRITIC ...] markers
    cleaned = _re.sub(r'\[OBSERVER-CRITIC[^\]]*\]', '', cleaned)

    # [SYSTEM_INSTRUCTION]: ... lines
    cleaned = _re.sub(r'\[SYSTEM_INSTRUCTION\]:.*?(?:\n|$)', '', cleaned)

    # Individual tape cells [NNN,N,N] MEMORY: ...
    cleaned = _re.sub(r'\s*\[?\d+,\d+,\d+\]?\s*MEMORY:.*?(?:\n|$)', '', cleaned)

    # Stray >> markers from tape head
    cleaned = _re.sub(r'^>>\s*', '', cleaned, flags=_re.MULTILINE)

    # Clean up whitespace
    cleaned = cleaned.strip()
    cleaned = _re.sub(r'\n{3,}', '\n\n', cleaned)

    if not cleaned:
        return "I processed that internally."
    return cleaned


@dataclass
class ContextObject:
    """All context gathered from the 5-tier recall."""
    conversation_history: str
    vector_memories: List[str]
    graph_context: List[str]
    lessons: List[str]
    foundation_facts: str
    tape_view: str = ""
    reconciliation_notes: str = ""


class BrainOrchestrator:
    """
    The Central Nervous System — coordinates all brain modules
    and the Five-Tier Memory System.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        model: str = "gemma3:4b",
        context_stream: Optional[ContextStream] = None,
        vector_store: Optional[VectorStore] = None,
        lesson_manager: Optional[LessonManager] = None,
        tape_machine: Optional[TapeMachine] = None,
        reconciler: Optional[CrossTierReconciler] = None,
        scope_manager: Optional[ScopeManager] = None,
        amygdala: Optional[Amygdala] = None,
        salience: Optional[SalienceScorer] = None,
        habit_cache: Optional[HabitCache] = None,
        neurotransmitters: Optional[NeurotransmitterState] = None,
        predictive_model: Optional[PredictiveModel] = None,
        dmn: Optional[DefaultModeNetwork] = None,
        circadian: Optional[CircadianProfile] = None,
        neuroplasticity: Optional[AutonomousNeuroplasticity] = None,
    ):
        self.kg = kg
        self.model = model

        # ─── Five-Tier Memory System ───────────────────────────
        self.context_stream = context_stream or ContextStream()
        self.vector_store = vector_store or VectorStore(model=model)
        self.lessons = lesson_manager or LessonManager()

        # ─── Phase 2 Systems ───────────────────────────────────
        self.tape = tape_machine or TapeMachine()
        self.reconciler = reconciler or CrossTierReconciler(model=model)
        self.scope_manager = scope_manager or ScopeManager()

        # ─── Brain Systems ─────────────────────────────────────
        self.amygdala = amygdala or Amygdala()
        self.salience = salience or SalienceScorer(attention_budget=10)
        self.habit_cache = habit_cache or HabitCache(threshold=15)
        self.nt = neurotransmitters or NeurotransmitterState()
        self.predictive_model = predictive_model or PredictiveModel(kg, model=model)
        self.dmn = dmn or DefaultModeNetwork(kg, model=model)
        self.circadian = circadian or CircadianProfile()
        self.neuroplasticity = neuroplasticity or AutonomousNeuroplasticity(
            kg, model=model, amygdala=self.amygdala
        )

        # ─── Observer-Critic (pre-response audit) ──────────────
        from neuroform.brain.observer_critic import ObserverCritic
        self.observer_critic = ObserverCritic(model=model)

        # Wire OllamaClient
        from neuroform.memory.working_memory import WorkingMemory
        self._wm_compat = WorkingMemory(capacity=20)
        self.client = OllamaClient(
            kg, model=model,
            working_memory=self._wm_compat,
            amygdala=self.amygdala,
        )

        # State tracking
        self._last_user_message: Optional[str] = None
        self._message_count: int = 0
        self._last_active: float = time.time()

        logger.info("BrainOrchestrator initialized with Five-Tier Memory + Phase 2 Systems")

    def process(self, user_id: str, message: str,
                user_name: str = "Unknown",
                scope: str = "PUBLIC") -> str:
        """
        Full cognitive processing pipeline with 5-tier memory recall.
        scope is a ground-rule parameter — every data entry is tagged.
        """
        self._message_count += 1
        self._last_active = time.time()

        # ──── 1. Circadian → Neurotransmitter modulation ────
        self.circadian.apply_to_neurotransmitters(self.nt)

        # ──── 2. Evaluate previous prediction (if exists) ────
        if self.predictive_model.last_prediction and self._last_user_message:
            error = self.predictive_model.evaluate_error(
                self.predictive_model.last_prediction, message
            )
            sources = self.predictive_model.last_context_sources
            signals = self.predictive_model.generate_feedback_signal(error, sources)
            if signals:
                self._apply_feedback_signals(signals)

        # ──── 3. RECALL — Gather context from all 5 tiers ────
        context = self._recall(message, user_id, user_name, scope=scope)

        # ──── 4. Check Habit Cache ────
        habit_key = self._compute_habit_key(message)
        cached = self.habit_cache.get_cached_response(habit_key)
        if cached:
            logger.info(f"Habit cache hit — bypassing LLM")
            self._observe(user_id, message, cached, user_name, scope=scope)
            self._last_user_message = message
            return cached

        # ──── 5. Predict user intent ────
        try:
            history_str = self.context_stream.get_context(max_turns=4)
            prediction = self.predictive_model.predict_intent(
                context.foundation_facts, history_str
            )
        except Exception as e:
            logger.warning(f"Prediction skipped: {e}")

        # ──── 6. LLM Inference (with 5-tier context and effectors) ────
        tiered_ctx = self._format_tiered_context(context)
        response = self._execute_inference_with_tools(user_id, message, scope, tiered_ctx, user_name=user_name)

        # ──── 7. Observe — persist to all tiers (scoped) ────
        self._observe(user_id, message, response, user_name, scope=scope)

        # ──── 8. Record habit (only substantive responses) ────
        if len(response) > 50:
            self.habit_cache.record_invocation(habit_key, response)

        # ──── 9. NT modulation from sentiment ────
        sentiment = self._estimate_sentiment(message)
        self.nt.modulate_from_sentiment(sentiment)

        # ──── 10. Auto-compaction check ────
        self._check_compaction(user_id, scope)

        self._last_user_message = message
        return response

    def _execute_inference_with_tools(self, user_id: str, message: str, scope: str,
                                       tiered_ctx: str, user_name: str = "Unknown") -> str:
        """Handles multi-turn inference for native Python tool execution via Ollama."""
        import os
        owner_env = os.environ.get("DISCORD_OWNER_ID", "")
        owner_ids = [uid.strip() for uid in owner_env.split(",")] if owner_env else []
        # Grant owner privileges to the autonomous daemon
        if user_id == "SYSTEM":
            owner_ids.append("SYSTEM")
            
        is_owner = bool(user_id in owner_ids)
        
        # Three-Tier System Prompt: Kernel + Identity + Perception HUD
        from neuroform.prompts.prompt_engine import assemble
        system_prompt = assemble(self, user_id, scope=scope, user_name=user_name)
        tool_instructions = system_prompt + "\n\n" + tool_registry.get_prompt_instructions(is_owner)
        
        tools = tool_registry.get_schemas(is_owner)
        
        MAX_TOOL_LOOPS = 5
        tool_outputs_this_turn = []  # Track for Observer-Critic
        conversation = [
            {"role": "system", "content": tool_instructions},
            {"role": "user", "content": f"{tiered_ctx}\n\nUSER MESSAGE:\n{message}"}
        ]
        
        try:
            import ollama
            import re
            from neuroform.tools.parser import parse_tool_args
            
            tool_pattern = re.compile(r"\[TOOL:\s*(\w+)\((.*?)\)\s*\]", re.DOTALL)
            
            for _ in range(MAX_TOOL_LOOPS):
                # We DO NOT pass `tools=tools` natively, avoiding 400 errors on strict models.
                response = ollama.chat(
                    model=self.model,
                    messages=conversation
                )
                
                msg = response.get("message", {})
                content = msg.get("content", "")
                
                tool_matches = tool_pattern.findall(content)
                if tool_matches:
                    conversation.append(msg)
                    
                    for func_name, args_str in tool_matches:
                        args = parse_tool_args(args_str)
                        logger.info(f"Nero autonomous execution: {func_name}({args})")
                        result = tool_registry.execute(func_name, args, is_owner=is_owner)
                        tool_outputs_this_turn.append({"name": func_name, "output": result})
                        
                        # Provide the result back as a user observation, explicitly forcing a stop to break loops
                        conversation.append({
                            "role": "user",
                            "content": f"TOOL OBSERVATION FOR {func_name}:\n{result}\n\n[SYSTEM_INSTRUCTION]: You have received the tool result. DO NOT call any more tools. Provide your final plain-text response now."
                        })
                    
                    # Loop continues, re-prompting the LLM with the tool result appended
                    continue
                
                # If no tool calls, this is the final text response
                # Safety: strip any leaked tool syntax before returning
                final_response = sanitize_tool_calls(content)

                # ── Observer-Critic Audit ──
                audit = self.observer_critic.audit_response(
                    user_message=message,
                    bot_response=final_response,
                    tool_outputs=tool_outputs_this_turn,
                    conversation_context=tiered_ctx,
                )
                if not audit.allowed:
                    logger.warning(f"Observer-Critic BLOCKED: {audit.reason}")
                    # Retry once with guidance
                    conversation.append({"role": "assistant", "content": content})
                    conversation.append({
                        "role": "user",
                        "content": (
                            f"[OBSERVER-CRITIC REJECTED YOUR RESPONSE]\n"
                            f"Reason: {audit.reason}\n"
                            f"Guidance: {audit.guidance}\n"
                            f"Rewrite your response following the guidance above. "
                            f"Do not apologize for the error, just provide the corrected response."
                        )
                    })
                    retry = ollama.chat(model=self.model, messages=conversation)
                    retry_content = retry.get("message", {}).get("content", "")
                    return sanitize_tool_calls(retry_content) if retry_content else final_response

                return final_response
                
            return "Error: Exceeded maximum autonomous tool execution loops."
            
        except Exception as e:
            logger.error(f"Tool inference failed, falling back: {e}")
            # Fallback to standard 
            fallback_response = self.client.chat_with_memory(
                user_id, message,
                skip_context_fetch=True,
                tiered_context=tiered_ctx,
            )
            # Safety: strip any leaked tool syntax from fallback
            return sanitize_tool_calls(fallback_response)

    def _check_compaction(self, user_id: str, scope: str = "PUBLIC"):
        """Check if auto-compaction is needed and trigger it."""
        if not self.context_stream.needs_compaction:
            return

        logger.info("Auto-compaction threshold reached — compacting...")
        try:
            import asyncio
            from neuroform.memory.compaction import compact_context

            # Run compaction asynchronously (fire and forget)
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(compact_context(
                    context_stream=self.context_stream,
                    vector_store=self.vector_store,
                    llm_client=self.client,
                    model=self.model,
                    user_id=user_id,
                    scope=scope,
                ))
            else:
                loop.run_until_complete(compact_context(
                    context_stream=self.context_stream,
                    vector_store=self.vector_store,
                    llm_client=self.client,
                    model=self.model,
                    user_id=user_id,
                    scope=scope,
                ))
        except Exception as e:
            logger.error(f"Auto-compaction failed: {e}")

    def _recall(self, query: str, user_id: str,
                user_name: str = "Unknown",
                scope: str = "PUBLIC") -> ContextObject:
        """
        Retrieve context from all 5 memory tiers + reconciliation.
        Every query is scoped by user_id + scope.
        """
        # T1: ContextStream (conversation history)
        conversation = self.context_stream.get_context(
            target_scope=scope, user_id=user_id, max_turns=50
        )

        # T2: Vector Store (semantic search — scoped)
        vector_results = self.vector_store.retrieve(
            query, scope=scope, user_id=user_id, max_results=5
        )
        vector_texts = [r["text"] for r in vector_results]

        # T3: Knowledge Graph — scoped
        graph_context = []
        try:
            # Query by user name (how ErnOS finds "Maria")
            if user_name and user_name != "Unknown":
                name_ctx = self.kg.query_context(
                    user_name, layer=None,
                    user_id=user_id, scope=scope,
                )
                if name_ctx:
                    graph_context.extend(name_ctx)

            # Query by user ID node
            user_node = f"User_{user_id}"
            user_ctx = self.kg.query_context(
                user_node, layer=None,
                user_id=user_id, scope=scope,
            )
            if user_ctx:
                graph_context.extend(user_ctx)

            # Self-context (Nero's own identity — always PUBLIC)
            self_ctx = self.kg.query_context(
                "Nero", layer=None,
                user_id=user_id, scope=scope,
            )
            if self_ctx:
                graph_context.extend(self_ctx)
        except Exception as e:
            logger.error(f"Graph recall error: {e}")

        # T4: Lessons — scoped
        lessons = self.lessons.get_all_lessons(user_id=user_id, scope=scope)

        # Foundation Knowledge Injection
        foundation = build_foundation_context(self.kg, query)

        # T5: Tape Machine view
        tape_view = self.tape.get_view()

        # Cross-Tier Reconciliation (T4 > T3 > T2)
        kg_strings = []
        for fact in graph_context:
            if isinstance(fact, dict):
                kg_strings.append(
                    f"{fact.get('source','')} ({fact.get('relationship','')}) "
                    f"{fact.get('target','')}"
                )
            else:
                kg_strings.append(str(fact))

        reconciliation_notes = ""
        try:
            result = self.reconciler.reconcile(
                lessons=lessons,
                kg_facts=kg_strings,
                vector_texts=vector_texts,
            )
            if result.conflicts:
                notes = [f"[RECONCILIATION: {result.stats.get('conflicts', 0)} conflicts detected]"]
                for c in result.conflicts:
                    notes.append(f"  ⚠ {c.conflict_type}:{c.reason}")
                reconciliation_notes = "\n".join(notes)
        except Exception as e:
            logger.warning(f"Reconciliation skipped: {e}")

        return ContextObject(
            conversation_history=conversation,
            vector_memories=vector_texts,
            graph_context=graph_context,
            lessons=lessons,
            foundation_facts=foundation,
            tape_view=tape_view,
            reconciliation_notes=reconciliation_notes,
        )

    def _observe(self, user_id: str, user_msg: str, bot_msg: str,
                 user_name: str = "Unknown", scope: str = "PUBLIC"):
        """
        Record an interaction across all memory tiers.
        Every write is tagged with user_id + scope — ground rule.
        """
        # T1: ContextStream — scoped
        self.context_stream.add_turn(
            user_id=user_id,
            user_message=user_msg,
            bot_message=bot_msg,
            user_name=user_name,
            scope=scope,
        )

        # T2: Vector Store — scoped
        exchange = f"{user_name}: {user_msg}\nNero: {bot_msg[:200]}"
        self.vector_store.store(exchange, user_id=user_id, scope=scope)

        # T5: Tape Machine — skip writing PRIVATE internal thoughts to the shared tape
        if scope != "PRIVATE":
            x, y, z = self.tape.focus_pointer
            next_x = x + 1
            self.tape.op_seek((next_x, 0, 0))
            self.tape.op_write(f"[{user_name}] {user_msg[:100]} → {bot_msg[:100]}")

        # T3 + T4: Entity Extraction + Lesson Learning — scoped
        if len(user_msg.strip()) > 5:
            self._extract_facts(user_id, user_msg, user_name, scope=scope)

    def _extract_facts(self, user_id: str, user_msg: str,
                       user_name: str = "Unknown",
                       scope: str = "PUBLIC"):
        """
        Deterministic entity/fact extraction via a dedicated LLM call.
        All extracted data is tagged with user_id + scope — ground rule.
        """
        import json as _json

        prompt = f"""Extract facts from this user message. The user's name is "{user_name}" (user ID: {user_id}).

MESSAGE: "{user_msg}"

If the message contains factual statements about the user (name, role, preferences, relationships, identity, etc.), output a JSON object with two arrays:
1. "entities": Array of {{source, relation, target}} triples for the knowledge graph.
2. "lessons": Array of plain-text fact strings to remember forever.

Rules:
- Use the user's ACTUAL NAME (not "the user") as the source entity.
- Relations should be uppercase like IS_A, LIKES, WORKS_ON, HAS_ROLE, PREFERS, KNOWS, etc.
- Only extract CLEAR, EXPLICIT facts. Do NOT infer or speculate.
- If there are NO facts to extract, output: {{"entities": [], "lessons": []}}

Examples:
Input: "My name is Maria, I am your developer"
Output: {{"entities": [{{"source": "Maria", "relation": "IS_A", "target": "Developer"}}, {{"source": "Maria", "relation": "IS_DEVELOPER_OF", "target": "Nero"}}], "lessons": ["Maria is a developer", "Maria is the developer of Nero"]}}

Input: "hi how are you"
Output: {{"entities": [], "lessons": []}}

Respond with ONLY the JSON object, nothing else."""

        try:
            import ollama
            result = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": 512},  # Cap output length for extraction
            )
            raw = result.get("message", {}).get("content", "")

            # Strip markdown fences if present
            if "```json" in raw:
                raw = raw.split("```json")[-1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            data = _json.loads(raw)

            # T3: KG Entity Extraction — tagged with user_id + scope
            entities = data.get("entities", [])
            for ent in entities:
                source = ent.get("source", "")
                relation = ent.get("relation", "")
                target = ent.get("target", "")
                if source and relation and target:
                    self.kg.add_node("Entity", source, layer="SOCIAL",
                                    user_id=user_id, scope=scope)
                    self.kg.add_node("Entity", target, layer="SOCIAL",
                                    user_id=user_id, scope=scope)
                    self.kg.add_relationship(
                        source, relation, target, strength=2.0,
                        user_id=user_id, scope=scope,
                    )
                    logger.info(f"KG learned [{scope}]: {source} -{relation}-> {target}")

            # T4: Lesson Extraction — tagged with user_id + scope
            lessons = data.get("lessons", [])
            for lesson in lessons:
                if lesson and len(lesson) > 3:
                    self.lessons.add_lesson(lesson, user_id=user_id, scope=scope)
                    logger.info(f"Lesson learned [{scope}]: {lesson}")

        except _json.JSONDecodeError:
            logger.debug(f"Fact extraction: no valid JSON from LLM")
        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}")

    def _format_tiered_context(self, ctx: ContextObject) -> str:
        """Format all 5 tiers + reconciliation into a context string for the LLM."""
        parts = []

        # Foundation facts (highest priority)
        if ctx.foundation_facts:
            parts.append(ctx.foundation_facts)

        # Reconciliation warnings
        if ctx.reconciliation_notes:
            parts.append(ctx.reconciliation_notes)

        # Lessons (ground truth)
        if ctx.lessons:
            lesson_block = "\n[LESSONS — verified facts (treat as ground truth)]\n"
            for lesson in ctx.lessons:
                lesson_block += f"  • {lesson}\n"
            lesson_block += "[/LESSONS]\n"
            parts.append(lesson_block)

        # Knowledge Graph
        if ctx.graph_context:
            kg_block = "\n[KNOWLEDGE GRAPH — entity relationships]\n"
            for fact in ctx.graph_context[:10]:
                kg_block += f"  • {fact}\n"
            kg_block += "[/KNOWLEDGE GRAPH]\n"
            parts.append(kg_block)

        # Vector memories
        if ctx.vector_memories:
            vec_block = "\n[RELATED MEMORIES — semantic matches]\n"
            for mem in ctx.vector_memories[:5]:
                vec_block += f"  • {mem}\n"
            vec_block += "[/RELATED MEMORIES]\n"
            parts.append(vec_block)

        # Tape Machine state
        if ctx.tape_view:
            parts.append(f"\n{ctx.tape_view}\n")

        # Conversation history
        if ctx.conversation_history and ctx.conversation_history != "No conversation history.":
            parts.append(f"\n[CONVERSATION HISTORY]\n{ctx.conversation_history}\n[/CONVERSATION HISTORY]\n")

        return "\n".join(parts) if parts else "No prior context available."

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return a snapshot of all system states."""
        return {
            "message_count": self._message_count,
            "neurotransmitters": self.nt.to_dict(),
            "circadian": self.circadian.get_modulation(),
            "context_stream": self.context_stream.snapshot(),
            "vector_store": self.vector_store.snapshot(),
            "lessons": self.lessons.snapshot(),
            "habit_cache": self.habit_cache.snapshot(),
            "tape": self.tape.snapshot(),
            "reconciler": self.reconciler.snapshot(),
            "scope_manager": self.scope_manager.snapshot(),
            "last_prediction": self.predictive_model.last_prediction,
        }

    def _apply_feedback_signals(self, signals: List[Dict[str, Any]]):
        """Apply STRENGTHEN/DECAY signals from the predictive model."""
        for signal in signals:
            action = signal.get("action")
            target = signal.get("target", "")
            amount = signal.get("amount", 0.1)

            if not target or "_" not in target:
                continue

            parts = target.split("_", 1)
            source_name, target_name = parts[0], parts[1]

            try:
                with self.kg.driver.session() as session:
                    if action == "STRENGTHEN":
                        session.run("""
                            MATCH ({name: $source})-[r]->({name: $target})
                            SET r.strength = COALESCE(r.strength, 1.0) + $amount
                        """, source=source_name, target=target_name, amount=amount)
                    elif action == "DECAY":
                        session.run("""
                            MATCH ({name: $source})-[r]->({name: $target})
                            SET r.strength = COALESCE(r.strength, 1.0) - $amount
                        """, source=source_name, target=target_name, amount=amount)
            except Exception as e:
                logger.warning(f"Feedback signal failed: {e}")

    def _compute_habit_key(self, message: str) -> str:
        """Derive a habit key from the full normalized message."""
        normalized = " ".join(message.lower().strip().split())
        return normalized if normalized else "empty"

    @staticmethod
    def _estimate_sentiment(message: str) -> float:
        """Simple keyword-based sentiment estimation."""
        positive = {"love", "great", "happy", "thanks", "awesome", "wonderful",
                    "amazing", "enjoy", "excellent", "good", "like", "appreciate"}
        negative = {"hate", "sad", "angry", "terrible", "awful", "bad",
                    "frustrated", "annoyed", "upset", "disappointed", "horrible"}

        tokens = set(message.lower().split())
        pos_count = len(tokens & positive)
        neg_count = len(tokens & negative)

        if pos_count + neg_count == 0:
            return 0.0
        return (pos_count - neg_count) / (pos_count + neg_count)
