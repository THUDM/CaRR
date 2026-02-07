import os
import re
import aiohttp
import asyncio
import urllib.parse
from collections import Counter

def normalize_string(text: str) -> str:
    """Basic string normalization."""
    # Convert to lowercase and normalize whitespace
    text = text.lower().strip()
    #divide word and remove punctuation
    pattern = r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]"
    words = re.findall(pattern, text)
    # Replace multiple spaces with single space
    text = " ".join(words)
    return text

def word_ngrams(text: str, n: int) -> set:
    """Generate word-level n-grams from text."""
    words = text.split()
    if len(words) < n:
        return set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}

def contain_forbidden_str(text: str, forbidden_str: str, ngram_size: int = 13) -> bool:

    if not forbidden_str.strip() or not text.strip():
        return False

    normalized_text = normalize_string(text)
    normalized_forbidden = normalize_string(forbidden_str)
    
    # If the query is too short, perform exact matching directly
    if len(normalized_forbidden.split()) < ngram_size:
        return normalized_forbidden in normalized_text
    
    # Generate n-grams for the forbidden string
    forbidden_ngrams = word_ngrams(normalized_forbidden, ngram_size)
    
    if not forbidden_ngrams:
        return normalized_forbidden in normalized_text
    
    text_ngrams = word_ngrams(normalized_text, ngram_size)

    return len(forbidden_ngrams & text_ngrams) > 0

async def search(query, num=10, forbidden_strs=[], proxy=None, retry_times=3, serp_api_key=None):
    """
    Asynchronously performs a web search with a retry mechanism.

    Args:
        query (str): The search query.
        forbidden_strs (list, optional): A list of strings. If any are found in a result block, it is skipped.
        num (int, optional): Number of search results to fetch.
        proxy (str, optional): Proxy to use for the request.
        retry_times (int, optional): Number of retry attempts.
        serp_api_key (str): API key of Serp.

    Returns:
        tuple:
            - str: Formatted search results or an error message.
            - dict: Mapping from result index to URL.
    """
    if not serp_api_key:
        raise ValueError("api_key is required for SerpAPI")
    
    params = {
        "q": query,
        "num": num,
        "engine": "google",
        "api_key": serp_api_key,
    }
    serp_api_url = "https://serpapi.com/search.json"
    
    timeout = aiohttp.ClientTimeout(total=30)
    last_error = None

    for attempt in range(retry_times):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(serp_api_url, params=params, proxy=proxy, ssl=False) as resp:
                    try:
                        resp.raise_for_status()
                    except aiohttp.ClientResponseError as e:
                        error_message = f"HTTP Error {e.status}: {e.message}"
                        last_error = error_message
                        if attempt < retry_times - 1:
                            print(f"Request search failed on attempt {attempt + 1}/{retry_times}. Retrying... Reason: {error_message}. Query: {query}")
                        continue
                    
                    data = await resp.json()
                    organic_results = data.get("organic_results", [])
                    
                    results = []
                    idx2url = {}
                    for item in organic_results:
                        title = item.get('title','')
                        url = item.get('link', '')
                        description = item.get('snippet', '')
                        
                        block = title + '\n' + url + '\n' + description
                        if forbidden_strs and any(contain_forbidden_str(block, s) for s in forbidden_strs):
                            continue
                        
                        idx = len(results)
                        block = (
                            f"[{idx}] Title: {title}\n"
                            f"[{idx}] URL Source: {url}\n"
                            f"[{idx}] Description: {item.get('snippet','')}\n"
                        )
                            
                        idx2url[idx] = url
                        results.append(block)
                    
                    return '\n'.join(results).strip(), idx2url
            
        except asyncio.TimeoutError:
            error_message = f"Request timed out after {timeout.total} seconds."
            last_error = error_message
            if attempt < retry_times - 1:
                print(f"Request search failed on attempt {attempt + 1}/{retry_times}. Retrying... Reason: {error_message}. Query: {query}")
            continue
            
        except Exception as e:
            error_message = f"An unexpected error occurred: {type(e).__name__}: {e}"
            last_error = error_message
            if attempt < retry_times - 1:
                print(f"Request search failed on attempt {attempt + 1}/{retry_times}. Retrying... Reason: {error_message}. Query: {query}")
            continue

    final_message = f"Failed to fetch search results after {retry_times} attempts."
    if last_error:
        final_message += f" Last error: {last_error}"
    
    print(final_message)
    return final_message, {}


async def parse_url(url, forbidden_strs=[], proxy=None, retry_times=3, jina_api_key=None):
    """
    Asynchronously fetches and parses the content of a URL with retries.

    Args:
        url (str): Target URL.
        forbidden_strs (list, optional): Strings that invalidate the content if found.
        proxy (str, optional): Proxy to use.
        retry_times (int, optional): Number of retry attempts.
        jina_api_key (str): API key of Jina.

    Returns:
        str: Parsed content or an error message.
    """
    if not jina_api_key:
        raise ValueError("jina_api_key is required for Jina Reader API")
    
    url = str(url)
    # Normalize the URL
    if "https://r.jina.ai/" in url:
        url = url.replace("https://r.jina.ai/", "")
    if "http://r.jina.ai/" in url:
        url = url.replace("http://r.jina.ai/", "")
    if "view-source:" in url:
        url = url.replace("view-source:", "")
        
    jina_api_url = f"https://r.jina.ai/{url}"
    headers = {
        "Authorization": f"Bearer {jina_api_key}",
    }
    timeout = aiohttp.ClientTimeout(total=30)
    last_error = None

    for attempt in range(retry_times):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(jina_api_url, headers=headers, proxy=proxy, ssl=False) as resp:
                    try:
                        resp.raise_for_status()  
                    except aiohttp.ClientResponseError as e:
                        error_message = f"HTTP Error {e.status}: {e.message}"
                        last_error = error_message
                        if attempt < retry_times - 1:
                            print(f"Request parse failed on attempt {attempt + 1}/{retry_times}. Retrying... Reason: {error_message}. URL: {url}")
                        continue
                    
                    raw_text = await resp.text()
                    if forbidden_strs and any(contain_forbidden_str(raw_text, s) for s in forbidden_strs):
                        error_message = "Failed: Forbidden string found in content."
                        last_error = error_message
                        break

                    return raw_text

        except asyncio.TimeoutError:
            error_message = f"Request timed out after {timeout.total} seconds."
            last_error = error_message
            if attempt < retry_times - 1:
                print(f"Request parse failed on attempt {attempt + 1}/{retry_times}. Retrying... Reason: {error_message}. URL: {url}")
            continue
            
        except Exception as e:
            error_message = f"An unexpected error occurred: {type(e).__name__}: {e}"
            last_error = error_message
            if attempt < retry_times - 1:
                print(f"Request parse failed on attempt {attempt + 1}/{retry_times}. Retrying... Reason: {error_message}. URL: {url}")
            continue

    final_message = f"Failed to parse URL content after {retry_times} attempts."
    if last_error:
        final_message += f" Last error: {last_error}"
    
    print(final_message)
    return final_message

def find(
    pattern: str,
    parse_content: str,
    max_results: int = 50,
    context_length: int = 200,
    word_overlap_threshold: float = 0.8,
):
    """
    Search for exact or highly similar matches of `pattern` in a long text,
    and return surrounding context for each match.

    Args:
        pattern: Pattern string to search for
        parse_content: Long text to search within
        max_results: Maximum number of results
        context_length: Number of characters before and after each match
        word_overlap_threshold: Threshold for fuzzy word overlap matching

    Returns:
        list[str]: List of context snippets around each match
    """
    results = []
    if not pattern or not parse_content:
        return []

    ori_content = parse_content
    pattern_lower = pattern.lower()
    parse_content_lower = parse_content.lower()
    pattern_len = len(pattern_lower)

    # Exact match
    start = 0
    while start < len(parse_content_lower):
        idx = parse_content_lower.find(pattern_lower, start)
        if idx == -1:
            break
        results.append((idx, idx + len(pattern), 1.0))
        start = idx + 1
        if len(results) >= max_results:
            break

    # Fuzzy matching if exact matches are insufficient
    if len(results) < max_results and pattern_len >= 3:

        pattern_words = re.findall(r"\b[a-zA-Z0-9]+\b", pattern_lower)
        if not pattern_words:
            pass
        else:
            window_size = pattern_len + 10
            step_size = pattern_len + 5
            if step_size == 0:
                step_size = 1

            pos = 0
            while pos <= len(parse_content_lower) - window_size and len(results) < max_results:
                window_text = parse_content_lower[pos : pos + window_size]
                window_words = re.findall(r"\b[a-zA-Z0-9]+\b", window_text)

                pattern_word_count = Counter(pattern_words)
                matched = 0
                for word in window_words:
                    if pattern_word_count.get(word, 0) > 0:
                        matched += 1
                        pattern_word_count[word] -= 1

                overlap_ratio = matched / len(pattern_words)

                if overlap_ratio >= word_overlap_threshold:
                    is_too_close = False
                    for existing_start, _, _ in results:
                        if abs(existing_start - pos) < context_length:
                            is_too_close = True
                            break
                    if not is_too_close:
                        rough_start = pos
                        rough_end = pos + window_size
                        results.append((rough_start, rough_end, overlap_ratio))

                pos += step_size

    results.sort(key=lambda x: x[0])

    contexts = []
    for start, end, score in results:
        ctx_start = max(0, start - context_length)
        ctx_end = min(len(ori_content), end + context_length)
        context = ori_content[ctx_start:ctx_end].strip()
        contexts.append(context)

    return contexts[:max_results]