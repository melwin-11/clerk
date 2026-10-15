import asyncio
import os
from typing import Dict, List, Optional, Set

import discord
from config import discord_channel_id, discord_token
from rag import RAGEngine


def split_message_for_discord(text: str, max_length: int = 1900) -> List[str]:
    """Splits long text into chunks that fit within Discord's 2000-character limit."""
    if len(text) <= max_length:
        return [text]

    chunks = []
    lines = text.split("\n")
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + 1 > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # If a single line is longer than max_length, force split it
            while len(line) > max_length:
                chunks.append(line[:max_length])
                line = line[max_length:]
            current_chunk = line
        else:
            current_chunk = f"{current_chunk}\n{line}" if current_chunk else line

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


class ClerkDiscordBot(discord.Client):
    """
    Discord bot for Clerk Local-RAG.
    Responds only in the designated channel.
    Starts answering queries after '/query' and terminates session on '/stop'.
    """

    def __init__(
        self,
        rag_engine: Optional[RAGEngine] = None,
        target_channel_id: Optional[str] = discord_channel_id,
        *args,
        **kwargs,
    ):
        intents = kwargs.pop("intents", None)
        if intents is None:
            intents = discord.Intents.default()
            intents.message_content = True

        super().__init__(intents=intents, *args, **kwargs)

        self.rag_engine = rag_engine or RAGEngine()
        self.target_channel_id = int(target_channel_id) if target_channel_id else None
        # Track active user IDs currently in an interactive query session
        self.active_sessions: Set[int] = set()

    async def on_ready(self):
        print(f"[INFO] Discord bot logged in as {self.user} (ID: {self.user.id})")
        if self.target_channel_id:
            channel = self.get_channel(self.target_channel_id)
            channel_name = channel.name if channel else f"ID: {self.target_channel_id}"
            print(f"[INFO] Listening specifically on channel: {channel_name}")
        else:
            print("[INFO] No specific channel ID set. Listening on all accessible channels.")

    async def send_formatted_response(
        self,
        channel: discord.abc.Messageable,
        query_text: str,
        result: Dict[str, any],
    ):
        """Formats the RAG answer and source citations, sending them in safe chunks."""
        answer = result.get("answer", "No answer generated.")
        context_items = result.get("context_items", [])

        # Build citation footer
        sources_text = ""
        if context_items:
            sources_lines = []
            for item in context_items[:3]:
                page = item.get("page_number", "?")
                score = item.get("score", 0.0)
                sources_lines.append(f"• **Page {page}** (relevance score: `{score:.3f}`)")
            sources_text = "\n\n📚 **Sources from document:**\n" + "\n".join(sources_lines)

        full_reply = (
            f"**Query**: *{query_text}*\n\n"
            f"🤖 **Answer**:\n{answer}"
            f"{sources_text}\n\n"
            f"─" * 20 + "\n"
            f"💡 *Session active. Ask your next question, or type `/stop` to end the session.*"
        )

        chunks = split_message_for_discord(full_reply)
        for chunk in chunks:
            await channel.send(chunk)

    async def on_message(self, message: discord.Message):
        # Ignore messages sent by the bot itself
        if message.author == self.user:
            return

        # Restrict to specified channel if configured
        if self.target_channel_id and message.channel.id != self.target_channel_id:
            return

        content = message.content.strip()
        user_id = message.author.id

        # 1. Handle /stop command
        if content.lower() == "/stop":
            if user_id in self.active_sessions:
                self.active_sessions.remove(user_id)
                await message.channel.send(
                    f"👋 Session ended for {message.author.mention}. Type `/query <your question>` whenever you want to ask another question."
                )
            else:
                await message.channel.send(
                    f"ℹ️ {message.author.mention}, you do not have an active session. Type `/query <your question>` to start."
                )
            return

        # 2. Handle /query trigger
        if content.lower().startswith("/query"):
            # Mark session as active
            self.active_sessions.add(user_id)
            query_prompt = content[len("/query"):].strip()

            if not query_prompt:
                await message.channel.send(
                    f"✅ **Session started for {message.author.mention}!**\n"
                    f"Please ask your question regarding the document. Type `/stop` anytime to conclude."
                )
                return

            # Execute RAG query with typing indicator
            async with message.channel.typing():
                loop = asyncio.get_running_loop()
                try:
                    rag_result = await loop.run_in_executor(
                        None, self.rag_engine.query, query_prompt
                    )
                    await self.send_formatted_response(
                        message.channel, query_prompt, rag_result
                    )
                except Exception as e:
                    print(f"[ERROR] Failed to process query '{query_prompt}': {e}")
                    await message.channel.send(
                        f"⚠️ Sorry {message.author.mention}, an error occurred while generating the answer: `{e}`"
                    )
            return

        # 3. Handle messages within an active session
        if user_id in self.active_sessions:
            async with message.channel.typing():
                loop = asyncio.get_running_loop()
                try:
                    rag_result = await loop.run_in_executor(
                        None, self.rag_engine.query, content
                    )
                    await self.send_formatted_response(
                        message.channel, content, rag_result
                    )
                except Exception as e:
                    print(f"[ERROR] Failed to process message '{content}': {e}")
                    await message.channel.send(
                        f"⚠️ Sorry {message.author.mention}, an error occurred while generating the answer: `{e}`"
                    )


def start_bot(token: str = discord_token, channel_id: str = discord_channel_id):
    """Initializes and runs the Discord Bot."""
    if not token:
        raise ValueError("DISCORD_TOKEN is missing. Please set it in .env or pass it as an argument.")

    print(f"[INFO] Initializing Clerk Discord Bot...")
    bot = ClerkDiscordBot(target_channel_id=channel_id)
    bot.run(token)


if __name__ == "__main__":
    start_bot()
