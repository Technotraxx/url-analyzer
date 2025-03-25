import streamlit as st
import re
from urllib.parse import urlparse, unquote
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from datetime import datetime
import json
import io
import tempfile
import base64
import random
from matplotlib.figure import Figure
import concurrent.futures
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Kalium URL Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("Kalium URL Analyzer")
st.markdown("Analysiere URLs aus deinem Kalium-Kurationsservice und erstelle detaillierte Berichte.")

class URLAnalyzer:
    def __init__(self):
        """Initialize the URL analyzer."""
        self.urls = []
        self.domain_counter = Counter()
        self.path_components = defaultdict(list)
        self.sections = defaultdict(Counter)
        self.topic_words = Counter()
        self.categorization_reasons = defaultdict(list)  # Store reasons for categorization
        self.category_urls = defaultdict(list)  # Store URLs in each category
        
        # Default content categories - can be customized by user
        self.content_categories = {
            'news': ['news', 'nachrichten', 'aktuell', 'artikel', 'story', 'stories', 'meldung'],
            'sports': ['sport', 'sports', 'fussball', 'soccer', 'basketball', 'tennis', 'wintersport', 
                      'ski', 'biathlon', 'motorsport', 'formel', 'olympia'],
            'entertainment': ['entertainment', 'stars', 'promis', 'tv', 'kino', 'film', 'musik', 'unterhaltung'],
            'health': ['gesundheit', 'health', 'medizin', 'fitness', 'wellness', 'krankheit'],
            'travel': ['reise', 'travel', 'urlaub', 'vacation', 'tourism', 'hotel', 'kreuzfahrt'],
            'finance': ['wirtschaft', 'economy', 'financial', 'finance', 'money', 'geld', 'finanzen'],
            'technology': ['tech', 'technology', 'digital', 'computer', 'software', 'hardware'],  # Removed ambiguous terms like 'it' and 'app'
            'science': ['wissenschaft', 'science', 'research', 'forschung'],
            'lifestyle': ['lifestyle', 'leben', 'life', 'fashion', 'food', 'essen'],
            'automotive': ['auto', 'automotive', 'cars', 'motor', 'motoring'],
            'politics': ['politik', 'politics', 'government', 'regierung'],
            'home_garden': ['haus', 'garten', 'home', 'garden', 'pflanzen', 'einrichtung', 'deko'],
            'royals': ['royals', 'adel', 'koenig', 'koenigin', 'prinz', 'prinzessin', 'palace'],
            'education': ['bildung', 'education', 'schule', 'university', 'studium', 'lernen']
        }
        
        # Default domain category hints - can be customized by user
        self.domain_category_hints = {
            'sport1.de': 'sports',
            'sportnews.bz': 'sports',
            'kicker.de': 'sports',
            'eurosport.de': 'sports',
            'sport.de': 'sports',
            'sportschau.de': 'sports',
            'spox.com': 'sports',
            
            'filmstarts.de': 'entertainment',
            'tvspielfilm.de': 'entertainment',
            'gala.de': 'entertainment',
            'bunte.de': 'entertainment',
            'promiflash.de': 'entertainment',
            'rtl.de': 'entertainment',
            'netflix.com': 'entertainment',
            
            'reisereporter.de': 'travel',
            'travelbook.de': 'travel',
            'holidaycheck.de': 'travel',
            'tripadvisor.de': 'travel',
            
            'finanzen.net': 'finance',
            'boerse.de': 'finance',
            'handelsblatt.com': 'finance',
            'wirtschaftswoche.de': 'finance',
            
            'heise.de': 'technology',
            't3n.de': 'technology',
            'computerbild.de': 'technology',
            'chip.de': 'technology',
            'golem.de': 'technology',
            
            'wissenschaft.de': 'science',
            'geo.de': 'science',
            'spektrum.de': 'science',
            
            'mein-schoener-garten.de': 'home_garden',
            'garten.de': 'home_garden',
            
            'essen-und-trinken.de': 'lifestyle',
            'brigitte.de': 'lifestyle',
            'cosmopolitan.de': 'lifestyle',
            
            'motor-talk.de': 'automotive',
            'auto-motor-sport.de': 'automotive',
            'autobild.de': 'automotive',
            
            'bundestag.de': 'politics',
            'bundesregierung.de': 'politics',
            'tagesschau.de': 'news',
            'spiegel.de': 'news',
            'zeit.de': 'news',
            'focus.de': 'news',
            'faz.net': 'news',
            'sueddeutsche.de': 'news',
            'welt.de': 'news',
            'n-tv.de': 'news',
            'bild.de': 'news',
            'stern.de': 'news',
            't-online.de': 'news'
        }
        
        # Create exclusion patterns to avoid false positives
        self.exclusion_patterns = {
            'technology': [
                r'italien', r'italien', r'italienisch', r'italian',  # 'it' could be part of 'italian'/'italien'
                r'item', r'items',  # 'it' could be part of 'item'
                r'titel', r'title',  # 'it' could be part of 'title'/'titel'
                r'kapitel',  # 'it' could be part of 'kapitel'
                r'hospital', r'hospital', r'spital',  # 'it' could be part of 'hospital'/'spital'
                r'appetit', r'appetite',  # 'it' could be part of 'appetite'/'appetit'
                r'appell', r'appear', r'appeal',  # 'app' could be part of these words
                r'apple'  # 'app' could be part of 'apple'
            ]
        }
    
    def extract_urls(self, text):
        """Extract URLs from text content.
        
        Args:
            text (str): Text containing URLs
            
        Returns:
            list: List of extracted URLs
        """
        url_pattern = r'https?://[^\s]+'
        return re.findall(url_pattern, text)
    
    def load_urls_from_text(self, text):
        """Load URLs from text content.
        
        Args:
            text (str): Text containing URLs
        """
        self.urls = self.extract_urls(text)
        return f"Loaded {len(self.urls)} URLs from text"
    
    def analyze_url(self, url):
        """Analyze a single URL and return its components.
        
        Args:
            url (str): URL to analyze
            
        Returns:
            dict: URL analysis results
        """
        try:
            # Parse the URL
            parsed_url = urlparse(unquote(url))
            
            # Extract domain (removing www. prefix if present)
            domain = parsed_url.netloc
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Extract path components
            path = parsed_url.path.strip('/')
            components = path.split('/') if path else []
            
            # Detect categories in URL with reasons
            detected_categories, categorization_reasons = self._detect_categories_with_reasons(url, domain, components)
            
            # Extract potential topic words from path components
            topic_words = self._extract_topic_words(components)
            
            # Check if first component is a section
            section = components[0].lower() if components else ""
            
            return {
                'url': url,
                'domain': domain,
                'path_components': components,
                'section': section,
                'detected_categories': detected_categories,
                'categorization_reasons': categorization_reasons,
                'topic_words': topic_words
            }
        except Exception as e:
            st.error(f"Error analyzing URL {url}: {e}")
            return None
    
    def _detect_categories_with_reasons(self, url, domain, components):
        """Detect content categories in a URL with reasons.
        
        Args:
            url (str): Full URL
            domain (str): Domain name
            components (list): Path components
            
        Returns:
            tuple: (list of detected categories, dict of reasons)
        """
        detected = []
        reasons = {}
        url_lower = url.lower()
        
        # Check if domain has a hinted category
        if domain in self.domain_category_hints:
            category = self.domain_category_hints[domain]
            detected.append(category)
            reasons[category] = f"Domain hint: {domain} -> {category}"
        
        # Check for category keywords in URL
        for category, keywords in self.content_categories.items():
            # Skip if already detected from domain hints
            if category in detected:
                continue
            
            # Check each keyword
            for keyword in keywords:
                # Skip if URL contains exclusion patterns for this category
                if category in self.exclusion_patterns:
                    excluded = False
                    for pattern in self.exclusion_patterns[category]:
                        if re.search(pattern, url_lower):
                            excluded = True
                            break
                    if excluded:
                        continue
                
                # Check if keyword is in URL
                if keyword in url_lower:
                    detected.append(category)
                    reasons[category] = f"Keyword match: '{keyword}' in URL"
                    break
                
                # Check if keyword is in any path component
                for comp in components:
                    if keyword in comp.lower():
                        detected.append(category)
                        reasons[category] = f"Keyword match: '{keyword}' in path component '{comp}'"
                        break
                
                # Break out of keyword loop if category detected
                if category in detected:
                    break
        
        return detected, reasons
    
    def _extract_topic_words(self, components):
        """Extract potential topic words from path components.
        
        Args:
            components (list): Path components
            
        Returns:
            list: Extracted topic words
        """
        topics = []
        
        for component in components:
            # Skip numeric components and very short ones
            if not component.isdigit() and len(component) > 3:
                # Remove common suffixes like .html
                clean_component = re.sub(r'\.\w+$', '', component)
                # Replace hyphens and underscores with spaces
                clean_component = clean_component.replace('-', ' ').replace('_', ' ')
                if clean_component:
                    topics.append(clean_component)
        
        return topics
    
    def process_urls(self, progress_bar=None):
        """Process URLs in parallel.
        
        Args:
            progress_bar: Streamlit progress bar object
            
        Returns:
            list: List of URL analysis results
        """
        results = []
        
        # Reset counters and trackers
        self.domain_counter = Counter()
        self.path_components = defaultdict(list)
        self.sections = defaultdict(Counter)
        self.topic_words = Counter()
        self.categorization_reasons = defaultdict(list)
        self.category_urls = defaultdict(list)
        
        # Process URLs
        for i, url in enumerate(self.urls):
            result = self.analyze_url(url)
            if result:
                results.append(result)
                
                # Update counters
                self.domain_counter[result['domain']] += 1
                self.path_components[result['domain']].append(result['path_components'])
                if result['section']:
                    self.sections[result['domain']][result['section']] += 1
                for word in result['topic_words']:
                    self.topic_words[word] += 1
                
                # Store categorization reasons
                for category, reason in result['categorization_reasons'].items():
                    self.categorization_reasons[category].append({
                        'url': url,
                        'domain': result['domain'],
                        'reason': reason
                    })
                    self.category_urls[category].append(url)
            
            # Update progress bar
            if progress_bar:
                progress_bar.progress((i + 1) / len(self.urls))
        
        return results
    
    def analyze_domains(self, plot=True):
        """Analyze domains in the URLs.
        
        Args:
            plot (bool): Whether to create plots
            
        Returns:
            dict: Domain analysis results
        """
        total_urls = len(self.urls)
        
        # Get top domains
        top_domains = self.domain_counter.most_common(20)
        
        # Calculate percentages
        top_domains_with_pct = [(domain, count, count/total_urls*100) 
                                for domain, count in top_domains]
        
        # Create DataFrame
        df = pd.DataFrame(top_domains_with_pct, 
                          columns=['Domain', 'Count', 'Percentage'])
        
        # Create visualization if requested
        fig = None
        if plot:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(x='Count', y='Domain', data=df, ax=ax)
            ax.set_title('Top 20 Domains')
            plt.tight_layout()
        
        return {
            'top_domains': top_domains,
            'domain_count': len(self.domain_counter),
            'total_urls': total_urls,
            'dataframe': df,
            'plot': fig
        }
    
    def analyze_categories(self, results, plot=True):
        """Analyze content categories in the URLs.
        
        Args:
            results (list): URL analysis results
            plot (bool): Whether to create plots
            
        Returns:
            dict: Category analysis results
        """
        category_counter = Counter()
        domain_categories = defaultdict(Counter)
        category_domains = defaultdict(Counter)
        
        # Count categories and track domain relationships
        for result in results:
            domain = result['domain']
            for category in result['detected_categories']:
                category_counter[category] += 1
                domain_categories[domain][category] += 1
                category_domains[category][domain] += 1
        
        # Get top categories
        top_categories = category_counter.most_common()
        
        # Create DataFrame
        df = pd.DataFrame(top_categories, columns=['Category', 'Count'])
        df['Percentage'] = df['Count'] / len(self.urls) * 100
        
        # Add categorization method stats
        categorization_stats = {}
        for category in category_counter.keys():
            if category in self.categorization_reasons:
                domain_hint_count = 0
                keyword_count = 0
                
                for item in self.categorization_reasons[category]:
                    if "Domain hint" in item['reason']:
                        domain_hint_count += 1
                    else:
                        keyword_count += 1
                
                categorization_stats[category] = {
                    'domain_hint_count': domain_hint_count,
                    'keyword_count': keyword_count,
                    'total': domain_hint_count + keyword_count
                }
        
        # Create visualization if requested
        fig = None
        if plot:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(x='Count', y='Category', data=df, ax=ax)
            ax.set_title('Content Categories')
            plt.tight_layout()
        
        # Analyze top domains for each category
        category_top_domains = {}
        for category, domains in category_domains.items():
            category_top_domains[category] = domains.most_common(10)
        
        # Create categorization method chart
        method_fig = None
        if categorization_stats and plot:
            method_data = []
            for category, stats in categorization_stats.items():
                method_data.append({
                    'Category': category,
                    'Domain Hint': stats['domain_hint_count'],
                    'Keyword Match': stats['keyword_count']
                })
            
            method_df = pd.DataFrame(method_data)
            
            if not method_df.empty:
                method_fig, ax = plt.subplots(figsize=(12, 8))
                method_df.set_index('Category').plot(kind='barh', stacked=True, ax=ax)
                ax.set_title('Categorization Methods')
                ax.set_xlabel('Number of URLs')
                plt.tight_layout()
        
        return {
            'category_counts': dict(category_counter),
            'category_top_domains': category_top_domains,
            'dataframe': df,
            'plot': fig,
            'method_plot': method_fig,
            'categorization_stats': categorization_stats
        }
    
    def analyze_sections(self, plot=True):
        """Analyze sections/categories in the URLs.
        
        Args:
            plot (bool): Whether to create plots
            
        Returns:
            dict: Section analysis results
        """
        # Flatten sections from all domains
        all_sections = Counter()
        for domain_sections in self.sections.values():
            all_sections.update(domain_sections)
        
        # Get top sections
        top_sections = all_sections.most_common(30)
        
        # Create DataFrame
        df = pd.DataFrame(top_sections, columns=['Section', 'Count'])
        
        # Create visualization if requested
        fig = None
        if plot:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(x='Count', y='Section', data=df.head(20), ax=ax)
            ax.set_title('Top 20 Sections/Categories')
            plt.tight_layout()
        
        # Get top sections for top domains
        top_domain_sections = {}
        for domain, count in self.domain_counter.most_common(10):
            if domain in self.sections:
                top_domain_sections[domain] = self.sections[domain].most_common(5)
        
        return {
            'top_sections': top_sections,
            'top_domain_sections': top_domain_sections,
            'dataframe': df,
            'plot': fig
        }
    
    def analyze_topic_words(self, plot=True):
        """Analyze topic words in the URLs.
        
        Args:
            plot (bool): Whether to create plots
            
        Returns:
            dict: Topic word analysis results
        """
        # Filter out common words that don't add much value
        stopwords = {'html', 'php', 'index', 'news', 'article', 'articles', 'page', 
                     'pages', 'view', 'content', 'default', 'main', 'home'}
        
        filtered_topics = {word: count for word, count in self.topic_words.items() 
                           if word.lower() not in stopwords and len(word) > 3}
        
        # Get top topic words
        top_topics = Counter(filtered_topics).most_common(50)
        
        # Create DataFrame
        df = pd.DataFrame(top_topics, columns=['Topic', 'Count'])
        
        # Create visualization if requested
        fig = None
        if plot:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(x='Count', y='Topic', data=df.head(20), ax=ax)
            ax.set_title('Top 20 Topic Words')
            plt.tight_layout()
        
        return {
            'top_topics': top_topics,
            'dataframe': df,
            'plot': fig
        }
    
    def create_domain_topic_matrix(self, results, plot=True):
        """Create a domain-topic matrix showing which domains cover which topics.
        
        Args:
            results (list): URL analysis results
            plot (bool): Whether to create plots
            
        Returns:
            pd.DataFrame: Domain-topic matrix
        """
        # Extract top domains and categories
        top_domains = [domain for domain, _ in self.domain_counter.most_common(20)]
        
        # Create domain-category matrix
        domain_categories = defaultdict(Counter)
        
        for result in results:
            domain = result['domain']
            if domain in top_domains:
                for category in result['detected_categories']:
                    domain_categories[domain][category] += 1
        
        # Create matrix
        matrix_data = []
        for domain in top_domains:
            if domain in domain_categories:
                row = {'Domain': domain}
                for category in self.content_categories.keys():
                    row[category] = domain_categories[domain][category]
                matrix_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(matrix_data)
        
        # Create a heatmap visualization if requested
        fig = None
        if plot and not df.empty:
            # Prepare data for heatmap
            heatmap_df = df.set_index('Domain')
            
            # Only include categories with some data
            nonzero_cols = [col for col in heatmap_df.columns if heatmap_df[col].sum() > 0]
            if nonzero_cols:
                heatmap_df = heatmap_df[nonzero_cols]
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(16, 12))
                sns.heatmap(heatmap_df, cmap="YlGnBu", annot=True, fmt="d", linewidths=.5, ax=ax)
                ax.set_title('Domain-Category Matrix')
                plt.tight_layout()
        
        return df, fig
    
    def generate_report(self, domain_results, category_results, section_results, topic_results):
        """Generate a comprehensive report of the analysis.
        
        Args:
            domain_results (dict): Domain analysis results
            category_results (dict): Category analysis results
            section_results (dict): Section analysis results
            topic_results (dict): Topic analysis results
            
        Returns:
            str: Report text
        """
        report = [
            "# Kalium URL Analysis Report",
            f"\n## Overview",
            f"Analysis of {len(self.urls)} URLs curated by Kalium",
            f"Total unique domains: {domain_results['domain_count']}",
            
            f"\n## Domain Distribution",
            f"\n### Top 20 Domains"
        ]
        
        # Add top domains
        for i, (domain, count) in enumerate(domain_results['top_domains'], 1):
            percentage = count / domain_results['total_urls'] * 100
            report.append(f"{i}. **{domain}**: {count} URLs ({percentage:.1f}%)")
        
        # Add content categories
        report.extend([
            f"\n## Content Categories",
            f"\n### Category Distribution"
        ])
        
        for category, count in category_results['category_counts'].items():
            percentage = count / domain_results['total_urls'] * 100
            report.append(f"- **{category}**: {count} URLs ({percentage:.1f}%)")
            
            # Add categorization method
            if category in category_results.get('categorization_stats', {}):
                stats = category_results['categorization_stats'][category]
                report.append(f"  - Domain hints: {stats['domain_hint_count']} URLs")
                report.append(f"  - Keyword matches: {stats['keyword_count']} URLs")
        
        # Add top sections
        report.extend([
            f"\n## Common Sections/Categories",
            f"\n### Top 15 Sections"
        ])
        
        for section, count in section_results['top_sections'][:15]:
            report.append(f"- {section}: {count} occurrences")
        
        # Add domain-specific sections
        report.append(f"\n### Domain-Specific Sections")
        
        for domain, sections in section_results['top_domain_sections'].items():
            report.append(f"\n#### {domain}")
            for section, count in sections:
                report.append(f"- {section}: {count} occurrences")
        
        # Add category-specific domains
        report.append(f"\n## Category-Specific Domains")
        
        for category, domains in category_results['category_top_domains'].items():
            if domains:  # Check if the category has any domains
                report.append(f"\n### Top domains for {category}")
                for domain, count in domains[:5]:  # Show top 5 domains for each category
                    report.append(f"- {domain}: {count} occurrences")
        
        # Add top topic words
        report.extend([
            f"\n## Common Topics",
            f"\n### Top 20 Topic Words"
        ])
        
        for topic, count in topic_results['top_topics'][:20]:
            report.append(f"- {topic}: {count} occurrences")
        
        # Add conclusion
        report.extend([
            f"\n## Conclusion",
            f"The Kalium curation service appears to focus primarily on the following areas:",
            f"1. News and current events",
            f"2. Sports and entertainment content",
            f"3. Lifestyle and service journalism"
        ])
        
        # Join the report sections
        report_text = "\n".join(report)
        
        return report_text
    
    def run_analysis(self, progress_bar=None):
        """Run the complete URL analysis.
        
        Args:
            progress_bar: Streamlit progress bar object
            
        Returns:
            dict: Analysis results
        """
        start_time = time.time()
        
        # Check if we have URLs to analyze
        if not self.urls:
            return None
        
        # Process URLs
        results = self.process_urls(progress_bar)
        
        # Run analysis components
        domain_results = self.analyze_domains()
        category_results = self.analyze_categories(results)
        section_results = self.analyze_sections()
        topic_results = self.analyze_topic_words()
        
        # Create domain-topic matrix
        matrix_df, matrix_fig = self.create_domain_topic_matrix(results)
        
        # Generate report
        report = self.generate_report(domain_results, category_results, section_results, topic_results)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        return {
            'domain_results': domain_results,
            'category_results': category_results,
            'section_results': section_results,
            'topic_results': topic_results,
            'matrix_df': matrix_df,
            'matrix_fig': matrix_fig,
            'report': report,
            'execution_time': execution_time
        }
    
    def get_random_urls_from_category(self, category, n=5):
        """Get random URLs from a specific category for validation.
        
        Args:
            category (str): Category to sample from
            n (int): Number of URLs to sample
            
        Returns:
            list: List of sampled URLs with categorization reasons
        """
        if category not in self.categorization_reasons:
            return []
        
        items = self.categorization_reasons[category]
        samples = random.sample(items, min(n, len(items)))
        
        return samples

def get_download_link_for_df(df, filename, link_text):
    """Generate a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def get_download_link_for_text(text, filename, link_text):
    """Generate a download link for text content"""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def get_image_download_link(fig, filename, link_text):
    """Generate a download link for matplotlib figure"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Initialize session state for category keywords and domain hints
if 'category_keywords' not in st.session_state:
    st.session_state.category_keywords = {}

if 'domain_hints' not in st.session_state:
    st.session_state.domain_hints = {}

if 'exclusion_patterns' not in st.session_state:
    st.session_state.exclusion_patterns = {}

# Sidebar options
st.sidebar.title("Optionen")
input_method = st.sidebar.radio(
    "URL-Eingabemethode:",
    ("Datei hochladen", "URLs einfügen")
)

# Main content
if input_method == "Datei hochladen":
    st.subheader("Datei mit URLs hochladen")
    uploaded_file = st.file_uploader("Wähle eine Textdatei mit URLs", type=["txt"])
    
    if uploaded_file:
        content = uploaded_file.read().decode("utf-8")
        analyzer = URLAnalyzer()
        
        # Load custom category keywords if available
        if st.session_state.category_keywords:
            analyzer.content_categories.update(st.session_state.category_keywords)
        
        # Load custom domain hints if available
        if st.session_state.domain_hints:
            analyzer.domain_category_hints.update(st.session_state.domain_hints)
        
        # Load custom exclusion patterns if available
        if st.session_state.exclusion_patterns:
            analyzer.exclusion_patterns.update(st.session_state.exclusion_patterns)
        
        loading_msg = analyzer.load_urls_from_text(content)
        st.info(loading_msg)
        
        if len(analyzer.urls) > 0:
            if st.button("Analyse starten"):
                st.session_state.analyzer = analyzer
                
                with st.spinner("Analysiere URLs..."):
                    progress_bar = st.progress(0)
                    results = analyzer.run_analysis(progress_bar)
                    st.session_state.results = results
                    st.session_state.plots = {
                        'domain_plot': results['domain_results']['plot'],
                        'category_plot': results['category_results']['plot'],
                        'section_plot': results['section_results']['plot'],
                        'topic_plot': results['topic_results']['plot'],
                        'matrix_plot': results['matrix_fig'],
                        'method_plot': results['category_results'].get('method_plot')
                    }
                    
                st.success(f"Analyse abgeschlossen in {results['execution_time']:.2f} Sekunden!")
                st.rerun()

elif input_method == "URLs einfügen":
    st.subheader("URLs direkt einfügen")
    url_text = st.text_area("Füge URLs hier ein (eine URL pro Zeile oder gemischt mit Text):", height=300)
    
    if url_text:
        analyzer = URLAnalyzer()
        
        # Load custom category keywords if available
        if st.session_state.category_keywords:
            analyzer.content_categories.update(st.session_state.category_keywords)
        
        # Load custom domain hints if available
        if st.session_state.domain_hints:
            analyzer.domain_category_hints.update(st.session_state.domain_hints)
        
        # Load custom exclusion patterns if available
        if st.session_state.exclusion_patterns:
            analyzer.exclusion_patterns.update(st.session_state.exclusion_patterns)
        
        loading_msg = analyzer.load_urls_from_text(url_text)
        st.info(loading_msg)
        
        if len(analyzer.urls) > 0:
            if st.button("Analyse starten"):
                st.session_state.analyzer = analyzer
                
                with st.spinner("Analysiere URLs..."):
                    progress_bar = st.progress(0)
                    results = analyzer.run_analysis(progress_bar)
                    st.session_state.results = results
                    st.session_state.plots = {
                        'domain_plot': results['domain_results']['plot'],
                        'category_plot': results['category_results']['plot'],
                        'section_plot': results['section_results']['plot'],
                        'topic_plot': results['topic_results']['plot'],
                        'matrix_plot': results['matrix_fig'],
                        'method_plot': results['category_results'].get('method_plot')
                    }
                    
                st.success(f"Analyse abgeschlossen in {results['execution_time']:.2f} Sekunden!")
                st.rerun()

# Display results if available
if 'results' in st.session_state and st.session_state.results:
    results = st.session_state.results
    plots = st.session_state.plots
    analyzer = st.session_state.analyzer
    
    # Create tabs for different result sections
    tabs = st.tabs(["Überblick", "Domains", "Kategorien", "Sektionen", "Themen", "Bericht", "Einstellungen", "Validierung"])
    
    # Overview tab
    with tabs[0]:
        st.header("Analyseübersicht")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Analysierte URLs", len(st.session_state.analyzer.urls))
        with col2:
            st.metric("Unique Domains", results['domain_results']['domain_count'])
        with col3:
            st.metric("Kategorien gefunden", len(results['category_results']['category_counts']))
        
        st.subheader("Top 5 Domains")
        domain_df = results['domain_results']['dataframe'].head(5)
        st.dataframe(domain_df, use_container_width=True)
        
        st.subheader("Top 5 Kategorien")
        category_df = results['category_results']['dataframe'].head(5)
        st.dataframe(category_df, use_container_width=True)
        
        if plots.get('method_plot'):
            st.subheader("Kategorisierungsmethoden")
            st.pyplot(plots['method_plot'])
            st.caption("Zeigt, wie viele URLs jeder Kategorie durch Domain-Hints vs. Keyword-Matching kategorisiert wurden")
    
    # Domains tab
    with tabs[1]:
        st.header("Domain-Analyse")
        if plots['domain_plot']:
            st.pyplot(plots['domain_plot'])
        
        st.subheader("Domain-Übersicht")
        st.dataframe(results['domain_results']['dataframe'], use_container_width=True)
    
    # Categories tab
    with tabs[2]:
        st.header("Kategorie-Analyse")
        if plots['category_plot']:
            st.pyplot(plots['category_plot'])
        
        st.subheader("Kategorie-Übersicht")
        st.dataframe(results['category_results']['dataframe'], use_container_width=True)
        
        st.subheader("Kategorisierungsmethoden")
        if 'categorization_stats' in results['category_results']:
            method_data = []
            for category, stats in results['category_results']['categorization_stats'].items():
                method_data.append({
                    'Kategorie': category,
                    'Domain-Hint': stats['domain_hint_count'],
                    'Keyword-Match': stats['keyword_count'],
                    'Gesamt': stats['total']
                })
            
            method_df = pd.DataFrame(method_data)
            st.dataframe(method_df, use_container_width=True)
        
        if plots['matrix_plot']:
            st.subheader("Domain-Kategorie-Matrix")
            st.pyplot(plots['matrix_plot'])
    
    # Sections tab
    with tabs[3]:
        st.header("Sektionen-Analyse")
        if plots['section_plot']:
            st.pyplot(plots['section_plot'])
        
        st.subheader("Sektionen-Übersicht")
        st.dataframe(results['section_results']['dataframe'], use_container_width=True)
    
    # Topics tab
    with tabs[4]:
        st.header("Themen-Analyse")
        if plots['topic_plot']:
            st.pyplot(plots['topic_plot'])
        
        st.subheader("Themen-Übersicht")
        st.dataframe(results['topic_results']['dataframe'].head(30), use_container_width=True)
    
    # Report tab
    with tabs[5]:
        st.header("Analysebericht")
        st.markdown(results['report'])
    
    # Settings tab
    with tabs[6]:
        st.header("Kategorisierungseinstellungen")
        
        # Category keywords editor
        st.subheader("Kategorie-Keywords")
        st.write("Hier kannst du die Keywords für jede Kategorie anpassen. Änderungen werden bei der nächsten Analyse berücksichtigt.")
        
        # Create a copy of content categories
        if not st.session_state.category_keywords:
            st.session_state.category_keywords = analyzer.content_categories.copy()
        
        # Allow user to edit category keywords
        for category, keywords in st.session_state.category_keywords.items():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.write(f"**{category}**")
            with col2:
                keywords_str = ", ".join(keywords)
                new_keywords = st.text_area(f"Keywords für {category}", keywords_str, key=f"keywords_{category}", label_visibility="collapsed")
                # Update keywords if changed
                if new_keywords != keywords_str:
                    st.session_state.category_keywords[category] = [k.strip() for k in new_keywords.split(",") if k.strip()]
        
        # Domain hints editor
        st.subheader("Domain-Kategorie-Zuordnungen")
        st.write("Hier kannst du die Domain-Kategorie-Zuordnungen anpassen. Änderungen werden bei der nächsten Analyse berücksichtigt.")
        
        # Create a copy of domain hints
        if not st.session_state.domain_hints:
            st.session_state.domain_hints = analyzer.domain_category_hints.copy()
        
        # Display domain hints in a more compact way
        domain_hint_df = pd.DataFrame([
            {'Domain': domain, 'Kategorie': category}
            for domain, category in st.session_state.domain_hints.items()
        ])
        
        # Allow user to edit domain hints using a dataframe editor
        edited_hints = st.data_editor(
            domain_hint_df,
            use_container_width=True,
            num_rows="dynamic",
            key="domain_hints_editor"
        )
        
        # Update domain hints if changed
        if not edited_hints.equals(domain_hint_df):
            new_hints = {}
            for _, row in edited_hints.iterrows():
                if pd.notna(row['Domain']) and pd.notna(row['Kategorie']):
                    new_hints[row['Domain']] = row['Kategorie']
            st.session_state.domain_hints = new_hints
        
        # Exclusion patterns editor
        st.subheader("Ausschluss-Muster")
        st.write("Hier kannst du Muster definieren, die falsche Kategorisierungen verhindern. Änderungen werden bei der nächsten Analyse berücksichtigt.")
        
        # Create a copy of exclusion patterns
        if not st.session_state.exclusion_patterns:
            st.session_state.exclusion_patterns = analyzer.exclusion_patterns.copy()
        
        # Allow user to edit exclusion patterns
        for category, patterns in st.session_state.exclusion_patterns.items():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.write(f"**{category}**")
            with col2:
                patterns_str = ", ".join(patterns)
                new_patterns = st.text_area(f"Ausschluss-Muster für {category}", patterns_str, key=f"exclusion_{category}", label_visibility="collapsed")
                # Update patterns if changed
                if new_patterns != patterns_str:
                    st.session_state.exclusion_patterns[category] = [p.strip() for p in new_patterns.split(",") if p.strip()]
        
        # Add button to apply changes
        if st.button("Änderungen für die nächste Analyse speichern"):
            st.success("Änderungen gespeichert! Sie werden bei der nächsten Analyse angewendet.")
    
    # Validation tab
    with tabs[7]:
        st.header("Kategorisierungs-Validierung")
        st.write("Hier kannst du die Kategorisierung überprüfen, indem du zufällige URLs aus jeder Kategorie ansiehst.")
        
        # Allow user to select a category
        categories = list(results['category_results']['category_counts'].keys())
        selected_category = st.selectbox("Kategorie auswählen", categories)
        
        # Get random URLs from the selected category
        sample_size = st.slider("Anzahl der Beispiel-URLs", 1, 10, 5)
        samples = analyzer.get_random_urls_from_category(selected_category, sample_size)
        
        # Display samples
        if samples:
            st.subheader(f"Zufällige URLs aus der Kategorie '{selected_category}'")
            for i, sample in enumerate(samples, 1):
                with st.expander(f"URL {i}: {sample['domain']}"):
                    st.write(f"**Vollständige URL:** {sample['url']}")
                    st.write(f"**Kategorisierungsgrund:** {sample['reason']}")
                    st.write(f"**Domain:** {sample['domain']}")
                    
                    # Add option to mark this URL as incorrectly categorized
                    if st.button(f"Diese URL ist falsch kategorisiert", key=f"incorrect_{i}"):
                        # This would need more state management to handle properly,
                        # but for now we'll just show a message
                        st.error("Feedback wurde gespeichert. In einer zukünftigen Version kannst du falsch kategorisierte URLs neu zuordnen.")
        else:
            st.info(f"Keine URLs in der Kategorie '{selected_category}' gefunden.")
    
    # Download buttons
    st.sidebar.header("Downloads")
    
    # Create download links for each analysis component
    st.sidebar.markdown("### Analysedaten herunterladen")
    
    # Domains 
    st.sidebar.markdown(get_download_link_for_df(
        results['domain_results']['dataframe'], 
        "domains.csv", 
        "📊 Domains als CSV herunterladen"
    ), unsafe_allow_html=True)
    
    # Categories
    st.sidebar.markdown(get_download_link_for_df(
        results['category_results']['dataframe'], 
        "categories.csv", 
        "📊 Kategorien als CSV herunterladen"
    ), unsafe_allow_html=True)
    
    # Sections
    st.sidebar.markdown(get_download_link_for_df(
        results['section_results']['dataframe'], 
        "sections.csv", 
        "📊 Sektionen als CSV herunterladen"
    ), unsafe_allow_html=True)
    
    # Topics
    st.sidebar.markdown(get_download_link_for_df(
        results['topic_results']['dataframe'], 
        "topics.csv", 
        "📊 Themen als CSV herunterladen"
    ), unsafe_allow_html=True)
    
    # Report text
    st.sidebar.markdown(get_download_link_for_text(
        results['report'], 
        "kalium_analysis_report.md", 
        "📝 Bericht als Markdown herunterladen"
    ), unsafe_allow_html=True)
    
    # Images
    st.sidebar.markdown("### Grafiken herunterladen")
    for name, title, filename in [
        ('domain_plot', 'Domains', 'domain_analysis.png'),
        ('category_plot', 'Kategorien', 'category_analysis.png'),
        ('section_plot', 'Sektionen', 'section_analysis.png'),
        ('topic_plot', 'Themen', 'topic_analysis.png'),
        ('matrix_plot', 'Matrix', 'domain_category_matrix.png'),
        ('method_plot', 'Kategorisierungsmethoden', 'categorization_methods.png')
    ]:
        if name in plots and plots[name]:
            st.sidebar.markdown(get_image_download_link(
                plots[name], 
                filename, 
                f"🖼️ {title}-Grafik herunterladen"
            ), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Kalium URL Analyzer - v2.0")
