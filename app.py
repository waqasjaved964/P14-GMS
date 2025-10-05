import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import re
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
from pydub import AudioSegment
import io
import tempfile
import base64
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure Tesseract path
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except:
    pass

# Page config
st.set_page_config(
    page_title="Grocery Expense Tracker",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .budget-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .warning-card {
        background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .item-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin-bottom: 0.5rem;
    }
    .pending-item {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .transaction-item {
        background-color: #f8f9fa;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
    }
    .transaction-item:hover {
        background-color: #e9ecef;
        transform: translateX(5px);
    }
    .developer-footer {
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        padding: 2rem;
        border-top: 1px solid #e0e0e0;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
if 'budget' not in st.session_state:
    st.session_state.budget = 50000
if 'pending_items' not in st.session_state:
    st.session_state.pending_items = []
if 'invoice_items' not in st.session_state:
    st.session_state.invoice_items = []

# Enhanced Pakistani grocery items with categories
GROCERY_ITEMS = {
    'wheat flour': {'urdu': 'آٹا', 'aliases': ['atta', 'aata', 'flour'], 'unit': 'kg', 'emoji': '🌾',
                    'category': 'Staples'},
    'rice': {'urdu': 'چاول', 'aliases': ['chawal', 'basmati'], 'unit': 'kg', 'emoji': '🍚', 'category': 'Staples'},
    'sugar': {'urdu': 'چینی', 'aliases': ['cheeni', 'chini'], 'unit': 'kg', 'emoji': '🧂', 'category': 'Staples'},
    'cooking oil': {'urdu': 'تیل', 'aliases': ['oil', 'tel', 'ghee'], 'unit': 'liter', 'emoji': '🛢️',
                    'category': 'Staples'},
    'milk': {'urdu': 'دودھ', 'aliases': ['doodh', 'dudh'], 'unit': 'liter', 'emoji': '🥛', 'category': 'Dairy'},
    'eggs': {'urdu': 'انڈے', 'aliases': ['anday', 'egg', 'anda'], 'unit': 'dozen', 'emoji': '🥚', 'category': 'Dairy'},
    'chicken': {'urdu': 'مرغی', 'aliases': ['murgi', 'murghi', 'broiler'], 'unit': 'kg', 'emoji': '🍗',
                'category': 'Meat'},
    'beef': {'urdu': 'گائے کا گوشت', 'aliases': ['gosht', 'meat', 'gai'], 'unit': 'kg', 'emoji': '🥩',
             'category': 'Meat'},
    'mutton': {'urdu': 'بکری کا گوشت', 'aliases': ['bakra', 'lamb', 'goat'], 'unit': 'kg', 'emoji': '🍖',
               'category': 'Meat'},
    'tomatoes': {'urdu': 'ٹماٹر', 'aliases': ['tamatar', 'tomato'], 'unit': 'kg', 'emoji': '🍅',
                 'category': 'Vegetables'},
    'onions': {'urdu': 'پیاز', 'aliases': ['piaz', 'pyaz', 'onion'], 'unit': 'kg', 'emoji': '🧅',
               'category': 'Vegetables'},
    'potatoes': {'urdu': 'آلو', 'aliases': ['aloo', 'alu', 'potato'], 'unit': 'kg', 'emoji': '🥔',
                 'category': 'Vegetables'},
    'garlic': {'urdu': 'لہسن', 'aliases': ['lehsan', 'lahsan'], 'unit': 'kg', 'emoji': '🧄', 'category': 'Vegetables'},
    'ginger': {'urdu': 'ادرک', 'aliases': ['adrak'], 'unit': 'kg', 'emoji': '🫚', 'category': 'Vegetables'},
    'green chilies': {'urdu': 'ہری مرچ', 'aliases': ['mirch', 'chili', 'chilli', 'hari mirch'], 'unit': 'kg',
                      'emoji': '🌶️', 'category': 'Vegetables'},
    'yogurt': {'urdu': 'دہی', 'aliases': ['dahi', 'curd'], 'unit': 'kg', 'emoji': '🥣', 'category': 'Dairy'},
    'bread': {'urdu': 'ڈبل روٹی', 'aliases': ['double roti', 'roti'], 'unit': 'piece', 'emoji': '🍞',
              'category': 'Bakery'},
    'tea': {'urdu': 'چائے', 'aliases': ['chai', 'chaey'], 'unit': 'packet', 'emoji': '🍵', 'category': 'Beverages'},
    'salt': {'urdu': 'نمک', 'aliases': ['namak'], 'unit': 'kg', 'emoji': '🧂', 'category': 'Staples'},
    'lentils': {'urdu': 'دال', 'aliases': ['daal', 'dal', 'lentil'], 'unit': 'kg', 'emoji': '🫘', 'category': 'Staples'},
    'flour': {'urdu': 'میدہ', 'aliases': ['maida'], 'unit': 'kg', 'emoji': '🌾', 'category': 'Staples'},
    'butter': {'urdu': 'مکھن', 'aliases': ['makhan'], 'unit': 'kg', 'emoji': '🧈', 'category': 'Dairy'},
    'cheese': {'urdu': 'پنیر', 'aliases': ['paneer'], 'unit': 'kg', 'emoji': '🧀', 'category': 'Dairy'},
    'fish': {'urdu': 'مچھلی', 'aliases': ['machli', 'machhli'], 'unit': 'kg', 'emoji': '🐟', 'category': 'Meat'},
    'bananas': {'urdu': 'کیلے', 'aliases': ['kelay', 'banana'], 'unit': 'dozen', 'emoji': '🍌', 'category': 'Fruits'},
    'apples': {'urdu': 'سیب', 'aliases': ['saib', 'apple'], 'unit': 'kg', 'emoji': '🍎', 'category': 'Fruits'},
    'oranges': {'urdu': 'مالٹا', 'aliases': ['malta', 'orange'], 'unit': 'kg', 'emoji': '🍊', 'category': 'Fruits'},
    'lemons': {'urdu': 'لیموں', 'aliases': ['lemon', 'lemo'], 'unit': 'kg', 'emoji': '🍋', 'category': 'Fruits'},
    'biscuits': {'urdu': 'بسکٹ', 'aliases': ['biscuit'], 'unit': 'packet', 'emoji': '🍪', 'category': 'Snacks'},
    'soft drinks': {'urdu': 'سافٹ ڈرنک', 'aliases': ['cold drink', 'pepsi', 'coke'], 'unit': 'bottle', 'emoji': '🥤',
                    'category': 'Beverages'},
    'water': {'urdu': 'پانی', 'aliases': ['mineral water', 'aqua'], 'unit': 'bottle', 'emoji': '💧',
              'category': 'Beverages'},
}

# File paths
DATA_FILE = 'transactions.json'
BUDGET_FILE = 'budget.json'


def load_data():
    """Load transactions from file"""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                st.session_state.transactions = json.load(f)
        if os.path.exists(BUDGET_FILE):
            with open(BUDGET_FILE, 'r') as f:
                st.session_state.budget = json.load(f).get('budget', 50000)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.session_state.transactions = []
        st.session_state.budget = 50000


def save_data():
    """Save transactions to file"""
    try:
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.transactions, f, ensure_ascii=False, indent=2)
        with open(BUDGET_FILE, 'w') as f:
            json.dump({'budget': st.session_state.budget}, f)
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")


def convert_audio_format(audio_bytes, target_format='wav'):
    """Convert audio to WAV format if needed"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_input:
            tmp_input.write(audio_bytes)
            input_path = tmp_input.name

        audio = AudioSegment.from_file(input_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_output:
            audio.export(tmp_output.name, format='wav')
            with open(tmp_output.name, 'rb') as f:
                wav_bytes = f.read()

        os.unlink(input_path)
        os.unlink(tmp_output.name)

        return wav_bytes
    except Exception as e:
        st.error(f"Audio conversion error: {str(e)}")
        return audio_bytes


def parse_voice_text(text):
    """Enhanced parser to extract items, quantities, and prices from voice input"""
    text = text.lower().strip()
    items = []

    number_words = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
        'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
        'half': 0.5, 'quarter': 0.25
    }

    words = text.split()
    for i, word in enumerate(words):
        if word in number_words:
            words[i] = str(number_words[word])
    text = ' '.join(words)

    patterns = [
        r'(\d+\.?\d*)\s*(kg|kilogram|liter|litre|dozen|piece|packet|bottle)?\s*([a-z\s]+?)\s+(\d+)\s*(rupees|rs|rp|rps)',
        r'([a-z\s]+?)\s+(\d+\.?\d*)\s*(kg|kilogram|liter|litre|dozen|piece|packet|bottle)?\s+(\d+)\s*(rupees|rs|rp|rps)',
        r'(\d+\.?\d*)\s*(kg|kilogram|liter|litre|dozen|piece|packet|bottle)?\s*([a-z\s]+?)\s+for\s+(\d+)',
        r'([a-z\s]+?)\s+(\d+)\s*(rupees|rs|rp|rps)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match) >= 4:
                try:
                    if 'for' in pattern:
                        quantity_str, unit, item_name, price_str = match[:4]
                    elif pattern.startswith('([a-z') and 'rupees' in pattern:
                        item_name, price_str = match[0], match[1]
                        quantity_str, unit = "1", "piece"
                    else:
                        quantity_str, unit, item_name, price_str, _ = match[:5]

                    quantity = float(quantity_str)
                    price = float(price_str)

                    item_name = item_name.strip()

                    if not unit or unit == '':
                        unit = 'piece'
                    elif unit in ['kg', 'kilogram']:
                        unit = 'kg'
                    elif unit in ['liter', 'litre']:
                        unit = 'liter'
                    elif unit == 'dozen':
                        unit = 'dozen'
                    else:
                        unit = unit

                    matched_item = None
                    item_data = None

                    for item_key, data in GROCERY_ITEMS.items():
                        all_names = [item_key] + data['aliases']
                        for name in all_names:
                            if name in item_name:
                                matched_item = item_key
                                item_data = data
                                break
                        if matched_item:
                            break

                    if matched_item:
                        items.append({
                            'name': matched_item.title(),
                            'quantity': quantity,
                            'unit': unit,
                            'price': price,
                            'emoji': item_data['emoji'],
                            'category': item_data.get('category', 'Other')
                        })
                except ValueError:
                    continue

    if not items:
        words = text.split()
        i = 0
        while i < len(words):
            word = words[i]
            for item_key, data in GROCERY_ITEMS.items():
                all_names = [item_key] + data['aliases']
                for name in all_names:
                    if name in word:
                        quantity = 1.0
                        price = 0.0

                        if i > 0 and words[i - 1].replace('.', '').isdigit():
                            quantity = float(words[i - 1])

                        if i + 1 < len(words) and words[i + 1].replace('.', '').isdigit():
                            price = float(words[i + 1])
                        elif i + 2 < len(words) and words[i + 2].replace('.', '').isdigit():
                            price = float(words[i + 2])

                        if price > 0:
                            items.append({
                                'name': item_key.title(),
                                'quantity': quantity,
                                'unit': data['unit'],
                                'price': price,
                                'emoji': data['emoji'],
                                'category': data.get('category', 'Other')
                            })
                        break
            i += 1

    return items


def transcribe_audio(audio_bytes, language='en-PK'):
    """Transcribe audio using Google Speech Recognition"""
    recognizer = sr.Recognizer()

    try:
        if not isinstance(audio_bytes, bytes):
            audio_bytes = audio_bytes.read()

        if audio_bytes[:4] != b'RIFF':
            audio_bytes = convert_audio_format(audio_bytes)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        with sr.AudioFile(tmp_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio, language=language, show_all=False)

        os.unlink(tmp_path)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Error with speech recognition service: {str(e)}"
    except Exception as e:
        return f"Error processing audio: {str(e)}"


def add_transaction(items):
    """Add transaction to the list"""
    for item in items:
        transaction = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'item': item['name'],
            'quantity': item['quantity'],
            'unit': item['unit'],
            'price': item['price'],
            'total': item['quantity'] * item['price'],
            'emoji': item['emoji'],
            'category': item.get('category', 'Other')
        }
        st.session_state.transactions.append(transaction)
    save_data()


def extract_text_from_image(image):
    """Extract text from image using OCR without OpenCV"""
    try:
        # Use PIL directly for basic OCR without OpenCV preprocessing
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image, config=custom_config)
        return text
    except Exception as e:
        return f"OCR Error: {str(e)}"


def parse_invoice_text(text):
    """Enhanced parser to extract grocery items and prices from invoice text"""
    lines = text.split('\n')
    items = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line or any(keyword in line.lower() for keyword in
                           ['invoice', 'date', 'total', 'thank you', '---', 'item', 'qty', 'unit', 'amount']):
            continue

        # Method 1: Multi-column format
        parsed_item = parse_multi_column_format(line)

        # Method 2: Single line with quantities and prices
        if not parsed_item:
            parsed_item = parse_single_line_format(line)

        # Method 3: Keyword-based extraction
        if not parsed_item:
            parsed_item = parse_using_keywords(line)

        if parsed_item and parsed_item['price'] > 0:
            items.append(parsed_item)

    return items


def parse_multi_column_format(line):
    """Parse invoice line in multi-column format"""
    try:
        # Split by multiple spaces
        parts = [p.strip() for p in re.split(r'\s{2,}', line) if p.strip()]

        if len(parts) >= 3:
            item_name = parts[0]

            # Extract quantity and unit from item name
            quantity, unit = extract_quantity_and_unit(item_name)

            # If quantity not found in name, try from parts
            if quantity == 1.0 and len(parts) > 1:
                try:
                    qty_from_part = float(parts[1])
                    if 0.1 <= qty_from_part <= 100:
                        quantity = qty_from_part
                except:
                    pass

            # Find prices in remaining parts
            price_candidates = []
            for part in parts[1:]:
                try:
                    clean_part = re.sub(r'[^\d.]', '', part)
                    if clean_part:
                        price = float(clean_part)
                        if 10 <= price <= 10000:
                            price_candidates.append(price)
                except:
                    continue

            if price_candidates:
                if len(price_candidates) >= 2:
                    unit_price = price_candidates[0]
                    total_price = price_candidates[-1]

                    # Verify logic
                    calculated_total = unit_price * quantity
                    if abs(total_price - calculated_total) > 100:
                        unit_price = price_candidates[-1] / quantity if quantity > 0 else price_candidates[-1]
                else:
                    unit_price = price_candidates[0]

                if unit_price > 0:
                    return create_invoice_item(item_name, quantity, unit, unit_price)

    except Exception as e:
        print(f"Error in multi-column: {e}")

    return None


def parse_single_line_format(line):
    """Parse single line format with embedded quantities and prices"""
    try:
        patterns = [
            r'([a-zA-Z\s]+?)\s+(\d+\.?\d*)\s*([a-zA-Z]*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)',
            r'([a-zA-Z\s]+?)\s+(\d+\.?\d*)\s*([a-zA-Z]*)\s+(\d+\.?\d*)',
            r'([a-zA-Z\s]+?)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)',
            r'([a-zA-Z\s]+?)\s+(\d+\.?\d*)\s+(\d+\.?\d*)',
        ]

        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                groups = match.groups()

                item_name = groups[0].strip()
                quantity = 1.0
                unit = 'piece'
                unit_price = 0

                # Extract quantity and unit
                if len(groups) >= 2:
                    try:
                        quantity = float(groups[1])
                    except:
                        pass

                # Determine unit from item name
                quantity_from_name, unit_from_name = extract_quantity_and_unit(item_name)
                if quantity_from_name != 1.0:
                    quantity = quantity_from_name
                    unit = unit_from_name

                # Find unit price
                price_index = 2
                if len(groups) > price_index:
                    try:
                        unit_price = float(groups[price_index])
                    except:
                        pass

                if unit_price > 0:
                    return create_invoice_item(item_name, quantity, unit, unit_price)

    except Exception as e:
        print(f"Error in single-line: {e}")

    return None


def parse_using_keywords(line):
    """Parse using known grocery item keywords"""
    try:
        line_lower = line.lower()

        item_patterns = [
            ('milk', 180, 'liter'),
            ('sugar', 110, 'kg'),
            ('flour', 120, 'kg'),
            ('atta', 120, 'kg'),
            ('tea', 300, 'packet'),
            ('chai', 300, 'packet'),
            ('rice', 250, 'kg'),
            ('chawal', 250, 'kg'),
            ('egg', 240, 'dozen'),
            ('anday', 240, 'dozen'),
            ('chicken', 450, 'kg'),
            ('murgi', 450, 'kg'),
            ('oil', 450, 'liter'),
            ('ghee', 450, 'liter'),
            ('bread', 50, 'piece'),
            ('roti', 50, 'piece'),
            ('tomato', 80, 'kg'),
            ('tamatar', 80, 'kg'),
            ('onion', 60, 'kg'),
            ('piaz', 60, 'kg'),
            ('potato', 40, 'kg'),
            ('aloo', 40, 'kg'),
        ]

        for keyword, typical_price, typical_unit in item_patterns:
            if keyword in line_lower:
                numbers = re.findall(r'\d+\.?\d*', line)
                if numbers:
                    quantity = 1.0
                    unit_price = typical_price

                    for num in numbers:
                        try:
                            num_float = float(num)
                            if 0.1 <= num_float <= 20:
                                quantity = num_float
                            elif 50 <= num_float <= 5000:
                                if abs(num_float - typical_price) < abs(unit_price - typical_price):
                                    unit_price = num_float
                        except:
                            continue

                    quantity_from_name, unit_from_name = extract_quantity_and_unit(line)
                    if quantity_from_name != 1.0:
                        quantity = quantity_from_name
                        typical_unit = unit_from_name

                    item_data = GROCERY_ITEMS.get(keyword, GROCERY_ITEMS['rice'])
                    return {
                        'name': keyword.title(),
                        'quantity': quantity,
                        'unit': typical_unit,
                        'price': unit_price,
                        'emoji': item_data['emoji'],
                        'category': item_data.get('category', 'Other'),
                        'total': quantity * unit_price
                    }

    except Exception as e:
        print(f"Error in keyword parsing: {e}")

    return None


def extract_quantity_and_unit(text):
    """Extract quantity and unit from text"""
    quantity = 1.0
    unit = 'piece'

    unit_patterns = [
        (r'(\d+\.?\d*)\s*(kg|kilogram|kilo)', 'kg'),
        (r'(\d+\.?\d*)\s*(l|litre|liter)', 'liter'),
        (r'(\d+\.?\d*)\s*(dozen|dz)', 'dozen'),
        (r'(\d+\.?\d*)\s*(g|gram)', 'gram'),
        (r'(\d+\.?\d*)\s*(ml)', 'ml'),
        (r'(\d+\.?\d*)\s*(packet|pkt)', 'packet'),
        (r'(\d+\.?\d*)\s*(piece|pc|pcs)', 'piece'),
        (r'(\d+\.?\d*)\s*(bottle)', 'bottle'),
    ]

    for pattern, standard_unit in unit_patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                quantity = float(match.group(1))
                unit = standard_unit
                break
            except:
                continue

    return quantity, unit


def create_invoice_item(item_name, quantity, unit, unit_price):
    """Create a standardized invoice item"""
    clean_name = re.sub(r'\d+\.?\d*\s*[a-zA-Z]*', '', item_name).strip()
    if not clean_name:
        clean_name = item_name

    matched_item = None
    item_data = None

    clean_name_lower = clean_name.lower()
    for item_key, data in GROCERY_ITEMS.items():
        all_names = [item_key] + data['aliases']
        for name in all_names:
            if name in clean_name_lower:
                matched_item = item_key
                item_data = data
                break
        if matched_item:
            break

    if not matched_item:
        for item_key, data in GROCERY_ITEMS.items():
            if any(alias in clean_name_lower for alias in [item_key] + data['aliases']):
                matched_item = item_key
                item_data = data
                break

    if not matched_item:
        matched_item = 'rice'
        item_data = GROCERY_ITEMS['rice']

    final_unit = item_data.get('unit', unit)

    return {
        'name': matched_item.title(),
        'quantity': quantity,
        'unit': final_unit,
        'price': unit_price,
        'emoji': item_data['emoji'],
        'category': item_data.get('category', 'Other'),
        'total': quantity * unit_price
    }


def process_invoice_file(uploaded_file):
    """Process uploaded invoice file (image or PDF)"""
    try:
        if uploaded_file.type == "application/pdf":
            images = convert_from_bytes(uploaded_file.read())
            all_text = ""
            for image in images:
                text = extract_text_from_image(image)
                all_text += text + "\n"
            return all_text
        else:
            image = Image.open(uploaded_file)
            return extract_text_from_image(image)
    except Exception as e:
        return f"Error processing file: {str(e)}"


def get_monthly_data():
    """Get current month's transactions"""
    current_month = datetime.now().strftime('%Y-%m')
    if st.session_state.transactions:
        df = pd.DataFrame(st.session_state.transactions)
        df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
        return df[df['month'] == current_month]
    return pd.DataFrame()


def get_weekly_data():
    """Get current week's transactions"""
    if st.session_state.transactions:
        df = pd.DataFrame(st.session_state.transactions)
        df['date'] = pd.to_datetime(df['date'])
        start_date = datetime.now() - timedelta(days=datetime.now().weekday())
        end_date = start_date + timedelta(days=6)
        return df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    return pd.DataFrame()


def get_category_spending(df):
    """Calculate spending by category"""
    if not df.empty and 'category' in df.columns:
        return df.groupby('category')['total'].sum().sort_values(ascending=False)
    return pd.Series()


def export_data():
    """Export transactions to CSV"""
    if st.session_state.transactions:
        df = pd.DataFrame(st.session_state.transactions)
        csv = df.to_csv(index=False)
        return csv
    return None


# Load data at startup
load_data()

# Header
st.markdown('<p class="main-header">🛒 Grocery Expense Tracker</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Developed by WAQAS JAVED</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings & Budget")

    # Budget Management
    st.subheader("💰 Budget Settings")
    new_budget = st.number_input(
        "Monthly Budget (Rs)",
        min_value=0,
        value=st.session_state.budget,
        step=1000,
        help="Set your monthly grocery budget"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Update Budget", use_container_width=True):
            st.session_state.budget = new_budget
            save_data()
            st.success("Budget updated!")
    with col2:
        if st.button("Reset Data", use_container_width=True):
            st.session_state.transactions = []
            save_data()
            st.success("Data reset!")

    st.divider()

    # Quick Stats
    st.subheader("📊 Quick Stats")
    monthly_df = get_monthly_data()
    weekly_df = get_weekly_data()

    total_spent_monthly = monthly_df['total'].sum() if not monthly_df.empty else 0
    total_spent_weekly = weekly_df['total'].sum() if not weekly_df.empty else 0
    remaining = st.session_state.budget - total_spent_monthly

    st.metric("Monthly Spent", f"Rs {total_spent_monthly:,.0f}")
    st.metric("Weekly Spent", f"Rs {total_spent_weekly:,.0f}")
    st.metric("Remaining", f"Rs {remaining:,.0f}")

    if remaining < 0:
        st.error("⚠️ Over Budget!")
    elif remaining < st.session_state.budget * 0.2:
        st.warning("⚠️ Low Budget Warning!")

    st.divider()

    # Data Export
    st.subheader("📁 Data Management")
    csv_data = export_data()
    if csv_data:
        st.download_button(
            label="Export to CSV",
            data=csv_data,
            file_name=f"grocery_expenses_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎤 Voice Entry", "✍️ Manual Entry", "📋 Transactions", "📊 Analytics", "🧾 Invoice Upload"
])

# Tab 1: Voice Recording
with tab1:
    st.header("🎤 Voice Entry")
    st.info(
        "💡 **Tip:** Speak clearly! Examples: '2 kg rice 300 rupees', '1 liter milk 250 rupees', '1 dozen eggs 400 rupees'")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎙️ Record Audio")

        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_size="2x",
            pause_threshold=3.0
        )

        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")

            if st.button("🎯 Process Recording", type="primary", use_container_width=True):
                with st.spinner("Processing your voice..."):
                    try:
                        text = transcribe_audio(audio_bytes, 'en-PK')

                        if text.startswith("Error") or text == "Could not understand audio":
                            st.error(f"❌ {text}")
                            st.info("🔊 Try speaking louder and clearer, or check your microphone permissions.")
                        else:
                            st.success(f"**🎯 Transcribed Text:** {text}")

                            items = parse_voice_text(text)

                            if items:
                                st.session_state.pending_items = items
                                st.success(f"✅ Found {len(items)} item(s)! Review below.")
                            else:
                                st.warning("No items detected. Try these examples:")
                                st.code("""
                                - "2 kg rice 300 rupees"
                                - "1 liter milk 250 rupees" 
                                - "1 dozen eggs 400 rupees"
                                - "1 kg chicken 600 rupees"
                                """)
                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")

    with col2:
        st.subheader("📤 Upload Voice Note")

        uploaded_audio = st.file_uploader(
            "Upload audio file",
            type=['wav', 'mp3', 'ogg', 'm4a', 'webm'],
            help="Upload a voice recording of your grocery purchase"
        )

        if uploaded_audio:
            st.audio(uploaded_audio.read(), format="audio/wav")
            uploaded_audio.seek(0)

            if st.button("🎯 Process Uploaded Audio", type="primary", use_container_width=True):
                with st.spinner("Processing uploaded audio..."):
                    try:
                        audio_bytes = uploaded_audio.read()
                        text = transcribe_audio(audio_bytes, 'en-PK')

                        if text.startswith("Error") or text == "Could not understand audio":
                            st.error(f"❌ {text}")
                        else:
                            st.success(f"**🎯 Transcribed Text:** {text}")

                            items = parse_voice_text(text)

                            if items:
                                st.session_state.pending_items = items
                                st.success(f"✅ Found {len(items)} item(s)! Review below.")
                            else:
                                st.warning("No items detected. Try a clearer recording.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    # Display pending items for review
    if st.session_state.pending_items:
        st.divider()
        st.subheader("📝 Review & Edit Detected Items")

        with st.form("review_items_form"):
            st.markdown("### Detected Items")

            for idx, item in enumerate(st.session_state.pending_items):
                with st.container():
                    st.markdown(f'<div class="pending-item">', unsafe_allow_html=True)

                    col1, col2, col3, col4 = st.columns([1, 2, 2, 2])

                    with col1:
                        st.write(f"**{item['emoji']}**")

                    with col2:
                        item['name'] = st.selectbox(
                            "Item Name",
                            options=[k.title() for k in GROCERY_ITEMS.keys()],
                            index=list(GROCERY_ITEMS.keys()).index(item['name'].lower()) if item[
                                                                                                'name'].lower() in GROCERY_ITEMS else 0,
                            key=f"name_{idx}"
                        )

                    with col3:
                        item_key = item['name'].lower()
                        unit = GROCERY_ITEMS.get(item_key, {}).get('unit', 'piece')
                        item['unit'] = unit

                        item['quantity'] = st.number_input(
                            f"Quantity",
                            min_value=0.1,
                            value=float(item['quantity']),
                            step=0.1,
                            key=f"qty_{idx}"
                        )
                        st.caption(f"Unit: {unit}")

                    with col4:
                        item['price'] = st.number_input(
                            "Price per Unit (Rs)",
                            min_value=0,
                            value=int(item['price']) if item['price'] > 0 else 100,
                            step=10,
                            key=f"price_{idx}"
                        )
                        total = item['quantity'] * item['price']
                        st.metric("Total", f"Rs {total:,.0f}")

                    st.markdown('</div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if st.form_submit_button("✅ Add All Items", type="primary", use_container_width=True):
                    for item in st.session_state.pending_items:
                        item_key = item['name'].lower()
                        if item_key in GROCERY_ITEMS:
                            item['category'] = GROCERY_ITEMS[item_key].get('category', 'Other')

                    add_transaction(st.session_state.pending_items)
                    st.session_state.pending_items = []
                    st.success("All items added successfully!")
                    st.rerun()

            with col2:
                if st.form_submit_button("🔄 Process Again", use_container_width=True):
                    st.rerun()

            with col3:
                if st.form_submit_button("❌ Cancel", use_container_width=True):
                    st.session_state.pending_items = []
                    st.rerun()

# Tab 2: Manual Entry
with tab2:
    st.header("✍️ Manual Entry")

    with st.form("manual_entry_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            item_name = st.selectbox(
                "Item Name",
                options=[k.title() for k in GROCERY_ITEMS.keys()],
                help="Select grocery item from the list"
            )

        with col2:
            item_key = item_name.lower()
            unit = GROCERY_ITEMS.get(item_key, {}).get('unit', 'piece')
            quantity = st.number_input(
                f"Quantity ({unit})",
                min_value=0.1,
                value=1.0,
                step=0.1,
                help="Enter the quantity purchased"
            )

        with col3:
            price = st.number_input(
                "Price per Unit (Rs)",
                min_value=0,
                value=100,
                step=10,
                help="Price per unit in Pakistani Rupees"
            )

        submitted = st.form_submit_button("Add Transaction", type="primary", use_container_width=True)

        if submitted:
            if price > 0:
                item_data = GROCERY_ITEMS.get(item_key, GROCERY_ITEMS['rice'])
                add_transaction([{
                    'name': item_name,
                    'quantity': quantity,
                    'unit': unit,
                    'price': price,
                    'emoji': item_data['emoji'],
                    'category': item_data.get('category', 'Other')
                }])
                st.success(f"✅ Added {quantity} {unit} of {item_name} for Rs {quantity * price:,.0f}")
                st.rerun()
            else:
                st.error("Please enter a valid price")

# Tab 3: Review Transactions
with tab3:
    st.header("📋 Transaction History")

    if st.session_state.transactions:
        df = pd.DataFrame(st.session_state.transactions)
        df = df.sort_values('date', ascending=False)

        # Summary Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", len(df))
        with col2:
            st.metric("Total Items", f"{df['quantity'].sum():.1f}")
        with col3:
            st.metric("Total Spent", f"Rs {df['total'].sum():,.0f}")
        with col4:
            st.metric("Avg per Transaction", f"Rs {df['total'].mean():,.0f}")

        st.divider()

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=pd.to_datetime(df['date'].min()).to_pydatetime() if not df.empty else datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=pd.to_datetime(df['date'].max()).to_pydatetime() if not df.empty else datetime.now()
            )
        with col3:
            selected_category = st.selectbox(
                "Filter by Category",
                options=['All'] + sorted(df['category'].unique().tolist()) if 'category' in df.columns else ['All']
            )

        # Filter dataframe
        filtered_df = df[
            (pd.to_datetime(df['date']) >= pd.to_datetime(start_date)) &
            (pd.to_datetime(df['date']) <= pd.to_datetime(end_date))
            ]

        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['category'] == selected_category]

        st.subheader(f"📄 Transactions ({len(filtered_df)})")

        # Display transactions
        for idx, row in filtered_df.iterrows():
            with st.container():
                st.markdown('<div class="transaction-item">', unsafe_allow_html=True)
                col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 2, 1])

                with col1:
                    st.write(f"**{row['emoji']}**")

                with col2:
                    st.write(f"**{row['item']}**")
                    st.caption(f"📅 {row['date']} at {row['time']}")
                    st.caption(f"🏷️ {row.get('category', 'N/A')}")

                with col3:
                    st.write(f"**{row['quantity']} {row['unit']}**")
                    st.caption(f"@ Rs {row['price']:,}")

                with col4:
                    st.write(f"**Rs {row['total']:,.0f}**")

                with col5:
                    if st.button("🗑️", key=f"del_{idx}"):
                        original_idx = st.session_state.transactions.index(row.to_dict())
                        st.session_state.transactions.pop(original_idx)
                        save_data()
                        st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)

        # Clear all button
        st.divider()
        if st.button("🗑️ Clear All Transactions", type="secondary", use_container_width=True):
            st.warning("This will delete ALL transactions. This action cannot be undone.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, Delete All", type="primary", use_container_width=True):
                    st.session_state.transactions = []
                    save_data()
                    st.rerun()
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.rerun()

    else:
        st.info("📝 No transactions yet. Start adding items using voice or manual entry!")

# Tab 4: Analytics
with tab4:
    st.header("📊 Spending Analytics")

    monthly_df = get_monthly_data()
    weekly_df = get_weekly_data()

    if not monthly_df.empty:
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)

        total_spent_monthly = monthly_df['total'].sum()
        total_spent_weekly = weekly_df['total'].sum() if not weekly_df.empty else 0
        remaining = st.session_state.budget - total_spent_monthly
        budget_used = (total_spent_monthly / st.session_state.budget) * 100 if st.session_state.budget > 0 else 0

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Monthly Spent</h3>
                <h2>Rs {total_spent_monthly:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            card_class = "success-card" if remaining >= 0 else "warning-card"
            st.markdown(f"""
            <div class="{card_class}">
                <h3>Remaining</h3>
                <h2>Rs {remaining:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="budget-card">
                <h3>Weekly Spent</h3>
                <h2>Rs {total_spent_weekly:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Budget Used</h3>
                <h2>{budget_used:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Spending by Category")
            category_spending = get_category_spending(monthly_df)
            if not category_spending.empty:
                fig = px.pie(
                    values=category_spending.values,
                    names=category_spending.index,
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No category data available")

        with col2:
            st.subheader("📈 Top Items This Month")
            item_spending = monthly_df.groupby('item')['total'].sum().sort_values(ascending=False).head(8)
            if not item_spending.empty:
                fig = px.bar(
                    x=item_spending.values,
                    y=item_spending.index,
                    orientation='h',
                    labels={'x': 'Amount (Rs)', 'y': 'Item'},
                    color=item_spending.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No item data available")

        # Daily spending trend
        st.subheader("📅 Daily Spending Trend")
        daily_spending = monthly_df.groupby('date')['total'].sum().reset_index()
        if not daily_spending.empty:
            fig = px.line(
                daily_spending,
                x='date',
                y='total',
                markers=True,
                labels={'total': 'Amount (Rs)', 'date': 'Date'},
                title="Daily Spending Pattern"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No daily spending data available")

        # Budget progress gauge
        st.subheader("💰 Budget Progress")
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=budget_used,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Budget Used (%)"},
            delta={'reference': 100},
            gauge={
                'axis': {'range': [None, 150]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "orange"},
                    {'range': [100, 150], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("📊 No data available for analytics. Add some transactions first!")

# Tab 5: Invoice Upload (FULLY FUNCTIONAL without OpenCV)
with tab5:
    st.header("🧾 Invoice Upload & OCR")
    st.info("📸 Upload grocery invoice images or PDFs for automatic data extraction!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Invoice")
        uploaded_invoice = st.file_uploader(
            "Choose invoice file",
            type=['png', 'jpg', 'jpeg', 'pdf'],
            help="Upload a clear image or PDF of your grocery invoice"
        )

        if uploaded_invoice:
            st.success(f"✅ File uploaded: {uploaded_invoice.name}")

            # Display preview
            if uploaded_invoice.type.startswith('image'):
                image = Image.open(uploaded_invoice)
                st.image(image, caption="Uploaded Invoice", use_column_width=True)
            else:
                st.info("📄 PDF invoice uploaded - text will be extracted from all pages")

            # Process invoice
            if st.button("🔍 Extract Items from Invoice", type="primary", use_container_width=True):
                with st.spinner("Processing invoice with OCR..."):
                    try:
                        # Extract text from invoice
                        extracted_text = process_invoice_file(uploaded_invoice)

                        if extracted_text and not extracted_text.startswith("Error"):
                            st.subheader("📝 Extracted Text")
                            st.text_area("OCR Results", extracted_text, height=200, key="ocr_output")

                            # Parse items from extracted text
                            invoice_items = parse_invoice_text(extracted_text)

                            if invoice_items:
                                st.session_state.invoice_items = invoice_items
                                st.success(f"✅ Found {len(invoice_items)} item(s) in the invoice!")

                                # Show summary
                                total_amount = sum(item['total'] for item in invoice_items)
                                st.metric("Total Invoice Amount", f"Rs {total_amount:,.0f}")
                            else:
                                st.warning("No items detected in the invoice text.")
                                st.info("Try uploading a clearer image or use manual entry below.")
                        else:
                            st.error("No text could be extracted from the invoice.")

                    except Exception as e:
                        st.error(f"Error processing invoice: {str(e)}")

    with col2:
        st.subheader("Manual Invoice Entry")
        st.info("Manually enter items from your invoice")

        with st.form("manual_invoice_form"):
            manual_items = []

            st.write("**Add Invoice Items**")
            num_items = st.number_input("Number of items", min_value=1, max_value=20, value=3)

            for i in range(num_items):
                col1, col2, col3 = st.columns(3)
                with col1:
                    item_name = st.selectbox(
                        f"Item {i + 1}",
                        options=[k.title() for k in GROCERY_ITEMS.keys()],
                        key=f"manual_inv_item_{i}"
                    )
                with col2:
                    quantity = st.number_input(
                        "Quantity",
                        min_value=0.1,
                        value=1.0,
                        step=0.1,
                        key=f"manual_inv_qty_{i}"
                    )
                with col3:
                    price = st.number_input(
                        "Price (Rs)",
                        min_value=0,
                        value=100,
                        step=10,
                        key=f"manual_inv_price_{i}"
                    )

                if item_name and price > 0:
                    item_key = item_name.lower()
                    item_data = GROCERY_ITEMS.get(item_key, {})
                    manual_items.append({
                        'name': item_name,
                        'quantity': quantity,
                        'unit': item_data.get('unit', 'piece'),
                        'price': price,
                        'emoji': item_data.get('emoji', '🛒'),
                        'category': item_data.get('category', 'Other')
                    })

            if st.form_submit_button("➕ Add Manual Invoice Items", type="primary", use_container_width=True):
                if manual_items:
                    st.session_state.invoice_items = manual_items
                    st.success(f"✅ {len(manual_items)} items ready for review!")
                else:
                    st.error("Please add at least one item with a valid price")

    # Display and manage invoice items
    if st.session_state.invoice_items:
        st.divider()
        st.subheader("📋 Review Invoice Items")

        total_invoice_amount = sum(item['quantity'] * item['price'] for item in st.session_state.invoice_items)

        st.metric("Total Invoice Amount", f"Rs {total_invoice_amount:,.0f}")

        # Display items with editing capability
        for idx, item in enumerate(st.session_state.invoice_items):
            with st.container():
                st.markdown('<div class="transaction-item">', unsafe_allow_html=True)

                col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 2])

                with col1:
                    st.write(f"**{item['emoji']}**")

                with col2:
                    new_name = st.selectbox(
                        "Item",
                        options=[k.title() for k in GROCERY_ITEMS.keys()],
                        index=list(GROCERY_ITEMS.keys()).index(item['name'].lower()) if item[
                                                                                            'name'].lower() in GROCERY_ITEMS else 0,
                        key=f"inv_name_{idx}"
                    )
                    item['name'] = new_name

                with col3:
                    item_key = item['name'].lower()
                    unit = GROCERY_ITEMS.get(item_key, {}).get('unit', 'piece')
                    item['unit'] = unit

                    item['quantity'] = st.number_input(
                        "Qty",
                        min_value=0.1,
                        value=float(item['quantity']),
                        step=0.1,
                        key=f"inv_qty_{idx}"
                    )
                    st.caption(f"Unit: {unit}")

                with col4:
                    item['price'] = st.number_input(
                        "Unit Price (Rs)",
                        min_value=0,
                        value=int(item['price']),
                        step=10,
                        key=f"inv_price_{idx}"
                    )

                with col5:
                    total = item['quantity'] * item['price']
                    item['total'] = total
                    st.metric("Total", f"Rs {total:,.0f}")

                st.markdown('</div>', unsafe_allow_html=True)

        # Action buttons for invoice items
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Add All to Transactions", type="primary", use_container_width=True):
                add_transaction(st.session_state.invoice_items)
                st.session_state.invoice_items = []
                st.success("All invoice items added to transactions!")
                st.rerun()

        with col2:
            if st.button("🔄 Process Again", use_container_width=True):
                st.session_state.invoice_items = []
                st.rerun()

        with col3:
            if st.button("❌ Clear Items", use_container_width=True):
                st.session_state.invoice_items = []
                st.rerun()

# Footer
st.divider()
st.markdown("""
<div class="developer-footer">
    <p><strong>Grocery Expense Tracker v4.0</strong></p>
    <p>Developed by <strong>WAQAS JAVED</strong></p>
    <p>Advanced Voice & Invoice Recognition System | © 2024 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)