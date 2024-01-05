import re
import unicodedata

# Define a function to remove non-textual elements
def remove_non_text(text):
    # Use a regular expression to match any non-alphanumeric characters
    pattern = re.compile(r"[^\w\s]")
    # Replace them with an empty string
    text = pattern.sub("", text)
    # Return the cleaned text
    return text

# Define a function to replace non-breaking spaces with regular spaces
def replace_nbsp(text):
    # Use the unicodedata module to normalize the text
    text = unicodedata.normalize("NFKD", text)
    # Replace any non-breaking spaces with regular spaces
    text = text.replace("\xa0", " ")
    # Return the cleaned text
    return text

# Define a function to remove unnecessary punctuation
def remove_punctuation(text):
    # Use a regular expression to match any dots or dashes
    pattern = re.compile(r"[\.\-]+")
    # Replace them with a single space
    text = pattern.sub(" ", text)
    # Return the cleaned text
    return text

# Define a function to normalize the text
def normalize_text(text):
    # Convert the text to lowercase
    #text = text.lower()
    # Remove any accents
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    # Replace any umlauts with their equivalent letters
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    # Return the normalized text
    return text

# Define a function to remove any URLs or links
def remove_urls(text):
    # Use a regular expression to match any URLs or links
    #pattern = re.compile(r"https?://\S+") 
    text = re.sub(r'http\S+', '', text)
    # Replace them with an empty string
    #text = pattern.sub("", text)
    # Return the cleaned text
    return text

def remove_linebreaks(text):
    # Replace any linebreaks with a single space
    text = text.replace("\n", " ")
    # Return the cleaned text
    return text

def clean_document(document,text_to_replace, replacement_text):
    # Loop through each document in the list
    for doc in document:
        # Get the text from the page_content attribute
        text = doc.page_content

        # Add a regular expression to remove everything before "Besprechungsergebnis"
        match = re.search(r"Besprechungsergebnis", text)
        if match:
            text = text[match.start():]

        # Apply the cleaning functions to the text
        # text = remove_non_text(text)
        # text = remove_punctuation(text)
        text = normalize_text(text)
        text = replace_nbsp(text)
        text = remove_urls(text)
        text = remove_linebreaks(text)
        text = text.replace(text_to_replace, replacement_text)
        text = text.replace("               ", replacement_text)

        # Update the page_content attribute with the cleaned text
        doc.page_content = text

    # Return the document list with the updated page_content
    return document

def clean_document_confluence(document):
    for doc in document:
        # Get the text from the page_content attribute
        input_string = doc.page_content
        # Entferne jeden "\xa0"
        input_string = input_string.replace("\xa0", "")

        # Entferne am Anfang: "true ZAHL ZAHL months"
        start_pattern = re.compile(r'true\s+\d+\s+\d+\s+months')
        input_string = re.sub(start_pattern, '', input_string)

        # Entferne am Ende: "#HEX #HEX #HEX ZAHL #HEX"
        end_pattern = re.compile(r'#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})\s+#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})\s+#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})\s+\d+\s+#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})')
        input_string = re.sub(end_pattern, '', input_string)

        # Entferne Sonderzeichen wie "---------"
        special_char_pattern = re.compile(r'[-]+')
        input_string = re.sub(special_char_pattern, '', input_string)

        # Entferne doppelte Leerzeichen
        input_string = re.sub(r'\s+', ' ', input_string)
        doc.page_content = input_string

    return document
    