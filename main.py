import asyncio
import re
import json
from typing import List, Dict, Any
import os
from PyCharacterAI import get_client
from PyCharacterAI.exceptions import SessionClosedError
from dotenv import load_dotenv
# RAG and AI Libraries
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_groq import ChatGroq


load_dotenv()



# Your Character.AI credentials
token = "8bffea7f61747077512e09269760e1db113b59e7"
character_id = "Xqr1QCxyTmqZsDKmiVq_yyAWHqXJ4SVpv7LvNTY450E"


# Add your Groq API key here (get from https://console.groq.com/keys)
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "your_groq_api_key_here"
doctor_jsonfile = "dataset.json"
# Suicide Prevention Resources - Verified Indian Helplines
SUICIDE_PREVENTION_RESOURCES = {
    "helplines": [
        {
            "name": "AASRA Mumbai",
            "phone": "022 2754 6669", 
            "availability": "24/7",
            "languages": "English, Hindi"
        },
        {
            "name": "Jeevan Aastha Helpline",
            "phone": "1800 233 3330",
            "availability": "24/7", 
            "languages": "Multiple Indian languages"
        },
        {
            "name": "Vandrevala Foundation",
            "phone": "9999 666 555",
            "availability": "24/7",
            "languages": "English, Hindi"
        },
        {
            "name": "1Life Crisis Support",
            "phone": "78930 78930",
            "availability": "24/7",
            "languages": "Hindi, English, Telugu, Tamil, Kannada, Malayalam"
        },
        {
            "name": "iCALL Helpline",
            "phone": "022 2556 3291",
            "availability": "Mon-Sat 8AM-10PM",
            "languages": "English, Hindi, Marathi"
        }
    ]
}


def load_doctors_data():
    """Load doctors from JSON file"""
    try:
        with open("dataset.json", 'r', encoding='utf-8') as f:
            doctors_data = json.load(f)
            print(f"✅ Successfully loaded {len(doctors_data)} counselors from dataset.json")
            if doctors_data:
                print(f"📋 Sample counselor: {doctors_data[0].get('name', 'Unknown')} - {doctors_data[0].get('specialization', 'Unknown')}")
            return doctors_data
    except FileNotFoundError:
        print(f"❌ Error: dataset.json not found!")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format: {e}")
        return []
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return []


doctors_data = load_doctors_data()


class SuicideIdeationDetector:
    """Class to detect suicide ideation in user messages"""

    def __init__(self):
        # Keywords and phrases associated with suicidal ideation
        self.suicide_keywords = [
            "want to die", "wanna die", "i want to die", "i wanna die",
            "kill myself", "end my life", "suicide", "suicidal thoughts",
            "no point in living", "life is meaningless", "can't take it anymore",
            "better off dead", "end it all", "take my own life", "harm myself",
            "don't want to live", "tired of living", "give up on life",
            "hopeless", "worthless", "no reason to live", "nothing to live for",
            "want to disappear", "wish i was dead", "rather be dead",
            "can't go on", "too much pain", "unbearable pain", "kms", "kys"
        ]

        self.crisis_patterns = [
            r"\b(want|wanna)\s+(to\s+)?die\b",
            r"\bkill\s+myself\b",
            r"\bend\s+my\s+life\b", 
            r"\bsuicide\b",
            r"\bsuicidal\b",
            r"\bharm\s+myself\b",
            r"\bbetter\s+off\s+dead\b",
            r"\bgive\s+up\s+on\s+life\b",
            r"\bcan't\s+take\s+it\b",
            r"\btoo\s+much\s+pain\b",
            r"\bkms\b",
            r"\bkys\b"
        ]

    def detect_suicide_ideation(self, message: str) -> bool:
        """Detect if message contains suicidal ideation"""
        message_lower = message.lower().strip()

        # Check for direct keyword matches
        for keyword in self.suicide_keywords:
            if keyword in message_lower:
                return True

        # Check for pattern matches  
        for pattern in self.crisis_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                return True

        return False


class MessageCensor:
    """Class to censor potentially violating messages using Groq"""

    def __init__(self, groq_client: Groq):
        self.groq_client = groq_client

        # Common terms that might violate C.AI TOS
        self.violation_indicators = [
            "kms", "kys", "kill myself", "kill yourself", "suicide", "suicidal",
            "self harm", "self-harm", "cut myself", "hurt myself", "end my life",
            "want to die", "wanna die", "better off dead", "take my own life",
            "harm myself", "overdose", "pills", "rope", "bridge", "jump off"
        ]

    def needs_censoring(self, message: str) -> bool:
        """Check if message contains potentially violating content"""
        message_lower = message.lower().strip()

        for indicator in self.violation_indicators:
            if indicator in message_lower:
                return True
        return False

    def censor_message(self, message: str) -> str:
        """Use Groq to censor the message while preserving meaning"""

        if not self.needs_censoring(message):
            return message  # No censoring needed

        censor_prompt = f"""
        You are a message sanitizer. Your job is to take a user message that might violate platform terms of service and rewrite it in a way that:

        1. PRESERVES the emotional content and meaning as much as possible
        2. REMOVES explicit self-harm language, suicide references, or terms that violate TOS
        3. KEEPS the same tone and urgency
        4. MAINTAINS the user's actual feelings and situation
        5. MAKES minimal changes - only what's necessary to avoid TOS violations

        Examples of good censoring:
        - "I want to kms" → "I'm feeling really hopeless and desperate"
        - "I should just kill myself" → "I feel like there's no way out of this pain"
        - "Maybe I should hurt myself" → "I'm having really dark thoughts right now"
        - "I'm going to end it all" → "I feel like I can't handle this anymore"

        The goal is to let the conversation continue naturally while avoiding platform violations.

        Original message: "{message}"

        Provide only the sanitized version, nothing else:"""
        """

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": censor_prompt}],
                temperature=0.1,
                max_tokens=150
            )

            censored = response.choices[0].message.content.strip()

            # Remove any quotes if Groq added them
            if (censored.startswith('"') and censored.endswith('"')) or (censored.startswith("'") and censored.endswith("'")):
                censored = censored[1:-1]

            return censored

        except Exception as e:
            print(f"❌ Error censoring message: {e}")
            # Fallback: basic keyword replacement
            return self.basic_censor_fallback(message)

    def basic_censor_fallback(self, message: str) -> str:
        """Fallback censoring method if Groq fails"""
        replacements = {
            "kms": "feeling really down",
            "kys": "you're struggling", 
            "kill myself": "hurt so much inside",
            "kill yourself": "you're in pain",
            "suicide": "giving up",
            "suicidal": "hopeless",
            "end my life": "can't take this pain",
            "want to die": "feel hopeless",
            "wanna die": "feel hopeless",
            "harm myself": "hurt inside",
            "hurt myself": "feel terrible"
        }

        censored = message.lower()
        for original, replacement in replacements.items():
            censored = censored.replace(original, replacement)

        # Try to maintain original capitalization style
        if message.isupper():
            return censored.upper()
        elif message.istitle():
            return censored.title()
        else:
            return censored


class DoctorRecommendationRAG:
    """RAG system for recommending mental health doctors"""

    def __init__(self, doctors_data: List[Dict], groq_api_key: str):
        self.doctors_data = doctors_data
        self.groq_client = Groq(api_key=groq_api_key)
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1
        )

        # Initialize FastEmbed embeddings
        self.embeddings = FastEmbedEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

        self.vectorstore = None
        self.setup_vectorstore()

    def setup_vectorstore(self):
        """Setup Chroma vectorstore with doctor data"""
        # Create documents from doctor data
        documents = []
        for doctor in self.doctors_data:
            # Format availability for better display
            availability_text = "Not specified"
            if doctor.get('availability'):
                available_days = []
                for day_info in doctor.get('availability', []):
                    day = day_info.get('day', '')
                    slots = day_info.get('slots', [])
                    if slots:
                        available_days.append(f"{day}: {', '.join(slots)}")
                if available_days:
                    availability_text = "; ".join(available_days)
            
            doc_text = f"""
            Dr. {doctor.get('name', 'N/A')}
            Specialization: {doctor.get('specialization', 'N/A')}
            Category: {doctor.get('category', 'N/A')}
            Experience: {doctor.get('experience', 'N/A')} years
            Phone: {doctor.get('phone', 'N/A')}
            Email: {doctor.get('email', 'N/A')}
            Consultation Fee: ₹{doctor.get('consultationFee', 'N/A')}
            Availability: {availability_text}
            Qualifications: {', '.join(doctor.get('qualifications', []))}
            Status: {doctor.get('status', 'N/A')}
            """

            documents.append(Document(
                page_content=doc_text,
                metadata={
                    "doctor_id": doctor.get('userId'),
                    "name": doctor.get('name'),
                    "specialization": doctor.get('specialization'),
                    "category": doctor.get('category'),
                    "experience": doctor.get('experience'),
                    "phone": doctor.get('phone'),
                    "consultationFee": doctor.get('consultationFee')
                }
            ))

        try:
            if not documents:
                print("❌ No documents to create vectorstore from!")
                self.vectorstore = None
                return
                
            print(f"📚 Creating vector database with {len(documents)} counselor profiles...")
            # Create vectorstore
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory="./doctor_chroma_db"
            )
            print("✅ Vector database created successfully!")
        except Exception as e:
            print(f"❌ Error creating vector database: {e}")
            self.vectorstore = None

    def find_relevant_doctors(self, query: str, k: int = 3) -> List[Document]:
        """Find relevant doctors based on query"""
        if not self.vectorstore:
            return []

        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []

    def generate_recommendation(self, user_situation: str) -> str:
        """Generate doctor recommendation using RAG"""
        print(f"🔍 Searching for counselors matching: {user_situation}")
        
        # Find relevant doctors
        relevant_docs = self.find_relevant_doctors(
            f"mental health psychiatrist psychologist depression anxiety therapy counseling crisis intervention {user_situation}",
            k=3
        )

        print(f"📋 Found {len(relevant_docs)} relevant counselors")
        
        if not relevant_docs:
            return "I couldn't find specific doctor recommendations at the moment. Please consider calling one of the crisis helplines for immediate support."

        # Prepare context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
        Based on the following doctor information, recommend the most suitable mental health professionals for someone experiencing emotional distress or mental health concerns.


        Available Doctors:
        {context}


        User situation: {user_situation}


        Please provide:
        1. Top 2-3 recommended doctors
        2. Why they are suitable for this situation
        3. Their contact information and availability
        4. Any special notes about their specialization


        Keep the response compassionate, professional, and focused on immediate help.
        Format the response clearly with doctor names, specializations, and contact details.
        """

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            return f"I encountered an issue generating recommendations. Please try calling one of the crisis helplines for immediate support."


class EnhancedChatBot:
    """Enhanced Character.AI chatbot with suicide prevention support and message censoring"""

    def __init__(self, token: str, character_id: str, groq_api_key: str):
        self.token = token
        self.character_id = character_id
        self.detector = SuicideIdeationDetector()
        self.rag_system = DoctorRecommendationRAG(doctors_data, groq_api_key)

        # Initialize Groq client for censoring
        self.groq_client = Groq(api_key=groq_api_key)
        self.censor = MessageCensor(self.groq_client)

        self.client = None
        self.chat = None
        self.me = None

        # Crisis tracking variables
        self.in_crisis_mode = False

    async def initialize(self):
        """Initialize the Character.AI client"""
        try:
            self.client = await get_client(token=self.token)
            self.me = await self.client.account.fetch_me()
            print(f'✅ Authenticated as @{self.me.username}')

            self.chat, greeting_message = await self.client.chat.create_chat(self.character_id)
            print(f"[{greeting_message.author_name}]: {greeting_message.get_primary_candidate().text}")

            print("\n" + "="*50)
            print("🤖 ENHANCED CHAT WITH MENTAL HEALTH SUPPORT & CENSORING")
            print("="*50)
            print("This chatbot includes:")
            print("• Immediate crisis support activation")
            print("• Smart message censoring to avoid TOS violations")
            print("• Professional mental health resources")
            print("="*50 + "\n")

        except Exception as e:
            print(f"❌ Error initializing Character.AI client: {e}")
            raise

    def show_crisis_resources(self):
        """Display crisis helplines"""
        print("\n" + "="*60)
        print("🆘 IMMEDIATE CRISIS SUPPORT AVAILABLE 🆘")
        print("="*60)
        print("If you're having thoughts of suicide or self-harm, please reach out immediately:")
        print()

        for helpline in SUICIDE_PREVENTION_RESOURCES["helplines"]:
            print(f"📞 {helpline['name']}: {helpline['phone']}")
            print(f"   Available: {helpline['availability']}")
            print(f"   Languages: {helpline['languages']}")
            print()

        print("These helplines are FREE, CONFIDENTIAL, and staffed by trained professionals.")
        print("They are available 24/7 and can provide immediate support and guidance.")
        print("="*60)

    def ask_for_preference(self) -> str:
        """Ask user if they want to talk to crisis helpline or get doctor recommendations"""
        print("\n" + "-"*40)
        print("SUPPORT OPTIONS:")
        print("-"*40)
        print("1. 📞 Call a crisis helpline now (RECOMMENDED for immediate help)")
        print("2. 👨‍⚕ Get recommendations for mental health professionals")
        print("3. ↩  Continue chatting normally")
        print("-"*40)

        while True:
            choice = input("Please enter 1, 2, or 3: ").strip()
            if choice in ['1', '2', '3']:
                return choice
            print("Please enter either 1, 2, or 3.")

    async def handle_crisis_message(self, message: str):
        """Handle detected crisis message"""
        # Show immediate crisis resources
        self.show_crisis_resources()

        # Ask for user preference
        choice = self.ask_for_preference()

        if choice == '1':
            print("\n✅ Please call one of the helplines above. They are available 24/7 and can provide immediate support.")
            print("Remember: You are not alone, and help is available. You matter. 💙")
            print("Consider saving these numbers in your phone for quick access.")

        elif choice == '2':
            print("\n🔍 Finding mental health professionals who can help you...")
            print("Please wait while I search our database...")

            recommendation = self.rag_system.generate_recommendation(message)
            print("\n" + "="*60)
            print("🏥 RECOMMENDED MENTAL HEALTH PROFESSIONALS")
            print("="*60)
            print(recommendation)
            print("\n💡 IMPORTANT REMINDERS:")
            print("• These are professional recommendations for ongoing support")
            print("• For immediate crisis support, please call the helplines above")
            print("• It's okay to reach out for help - it shows strength, not weakness")
            print("• Consider involving a trusted friend or family member in your care")
            print("="*60)

        else:  # choice == '3'
            print("\n💙 I'm here to listen. Remember that if you need immediate help,")
            print("   the crisis helplines above are always available.")


    async def run_chat(self):
        """Main chat loop with immediate crisis response and message censoring"""
        await self.initialize()

        try:
            while True:
                # Get user input
                print(f"\n[{self.me.name}]: ", end="")
                original_message = input()

                # Handle empty messages
                if not original_message.strip():
                    continue

                # STEP 1: Check for suicide ideation on ORIGINAL message (before censoring)
                crisis_detected = self.detector.detect_suicide_ideation(original_message)

                # STEP 2: Censor the message for Character.AI if needed
                censored_message = self.censor.censor_message(original_message)

                # Show censoring notification if message was censored
                if censored_message != original_message:
                    print(f"🔍 [Message sanitized for platform compatibility]")

                # STEP 3: Handle crisis detection (using original message context)
                if crisis_detected:
                    # IMMEDIATE CRISIS SUPPORT - Show options right away
                    if not self.in_crisis_mode:
                        print(f"\n🚨 I notice you're going through a difficult time. Let me connect you with professional help immediately.")
                        self.in_crisis_mode = True
                        await self.handle_crisis_message(original_message)

                        # Ask if they want to continue chatting
                        print(f"\n" + "-"*40)
                        continue_chat = input("Would you like to continue our conversation? (y/n): ").lower().strip()
                        if continue_chat not in ['y', 'yes']:
                            print("Take care of yourself. Remember, help is always available. 💙")
                            break
                        continue
                    else:
                        # Already in crisis mode, show resources again
                        print("\n🆘 You're still expressing concerning thoughts. Please consider the resources I shared.")
                        await self.handle_crisis_message(original_message)
                        continue

                else:
                    # No crisis detected - reset crisis mode if user is having normal conversation
                    if self.in_crisis_mode:
                        # Reset crisis mode if user is having normal conversation
                        self.in_crisis_mode = False

                # STEP 4: Send censored message to Character.AI (normal conversation)
                answer = await self.client.chat.send_message(
                    self.character_id, 
                    self.chat.chat_id, 
                    censored_message,  # Use censored version
                    streaming=True
                )

                # Handle Character.AI response (unchanged original functionality)
                printed_length = 0
                async for response in answer:
                    if printed_length == 0:
                        print(f"[{response.author_name}]: ", end="")

                    text = response.get_primary_candidate().text
                    print(text[printed_length:], end="")
                    printed_length = len(text)
                print()

        except KeyboardInterrupt:
            print("\n\n👋 Chat ended by user. Take care!")
        except SessionClosedError:
            print("Session closed. Bye!")
        except Exception as e:
            print(f"❌ An error occurred: {e}")
        finally:
            if self.client:
                await self.client.close_session()
                print("🔐 Session closed securely.")


async def main():
    """Main function"""
    # Check if Groq API key is set
    if GROQ_API_KEY == "your_groq_api_key_here":
        print("❌ ERROR: Please set your Groq API key!")
        print("1. Go to https://console.groq.com/keys")
        print("2. Create a new API key")
        print("3. Replace 'your_groq_api_key_here' with your actual API key")
        return

    print("🚀 Starting Enhanced Character.AI Bot with Mental Health Support & Smart Censoring...")
    print("⚡ Powered by Groq, LangChain, and FastEmbed")

    try:
        bot = EnhancedChatBot(token, character_id, GROQ_API_KEY)
        await bot.run_chat()
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")
        print("Please check your API keys and internet connection.")


if __name__ == "__main__":
    asyncio.run(main())
