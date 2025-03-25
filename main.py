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
from matplotlib.figure import Figure
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration
import markdown
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
        
        # Create content categories for classification
        self.content_categories = {
            'news': ['news', 'nachrichten', 'aktuell', 'artikel', 'story', 'stories', 'meldung'],
            'sports': ['sport', 'sports', 'fussball', 'soccer', 'basketball', 'tennis', 'wintersport', 
                      'ski', 'biathlon', 'motorsport', 'formel', 'olympia'],
            'entertainment': ['entertainment', 'stars', 'promis', 'tv', 'kino', 'film', 'musik', 'unterhaltung'],
            'health': ['gesundheit', 'health', 'medizin', 'fitness', 'wellness', 'krankheit'],
            'travel': ['reise', 'travel', 'urlaub', 'vacation', 'tourism', 'hotel', 'kreuzfahrt'],
            'finance': ['wirtschaft', 'economy', 'financial', 'finance', 'money', 'geld', 'finanzen'],
            'technology': ['tech', 'technology', 'digital', 'it', 'computer', 'software', 'hardware', 'app'],
            'science': ['wissenschaft', 'science', 'research', 'forschung'],
            'lifestyle': ['lifestyle', 'leben', 'life', 'fashion', 'food', 'essen'],
            'automotive': ['auto', 'automotive', 'cars', 'motor', 'motoring'],
            'politics': ['politik', 'politics', 'government', 'regierung'],
            'home_garden': ['haus', 'garten', 'home', 'garden', 'pflanzen', 'einrichtung', 'deko'],
            'royals': ['royals', 'adel', 'koenig', 'koenigin', 'prinz', 'prinzessin', 'palace'],
            'education': ['bildung', 'education', 'schule', 'university', 'studium', 'lernen']
        }
        
        # Create a mapping of domains to categories they likely belong to
        self.domain_category_hints = {
            'sport1.de': 'sports',
            'sportnews.bz': 'sports',
            'kicker.de': 'sports',
            'eurosport.de': 'sports',
            'filmstarts.de': 'entertainment',
            'tvspielfilm.de': 'entertainment',
            'gala.de': 'entertainment',
            'bunte.de': 'entertainment',
            'reisereporter.de': 'travel',
            'travelbook.de': 'travel',
            'finanzen.net': 'finance',
            'boerse.de': 'finance',
            'heise.de': 'technology',
            't3n.de': 'technology',
            'wissenschaft.de': 'science',
            'geo.de': 'science',
            'mein-schoener-garten.de': 'home_garden',
            'essen-und-trinken.de': 'lifestyle',
            'motor-talk.de': 'automotive',
            'auto-motor-sport.de': 'automotive',
            'bundestag.de': 'politics',
            'royals.de': 'royals'
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
            
            # Detect categories in URL
            detected_categories = self._detect_categories(url, domain, components)
            
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
                'topic_words': topic_words
            }
        except Exception as e:
            st.error(f"Error analyzing URL {url}: {e}")
            return None
    
    def _detect_categories(self, url, domain, components):
        """Detect content categories in a URL.
        
        Args:
            url (str): Full URL
            domain (str): Domain name
            components (list): Path components
            
        Returns:
            list: Detected categories
        """
        detected = []
        url_lower = url.lower()
        
        # Check if domain has a hinted category
        if domain in self.domain_category_hints:
            detected.append(self.domain_category_hints[domain])
        
        # Check for category keywords in URL
        for category, keywords in self.content_categories.items():
            # Skip if already detected from domain hints
            if category in detected:
                continue
                
            for keyword in keywords:
                if keyword in url_lower or any(keyword in comp.lower() for comp in components):
                    detected.append(category)
                    break
        
        return detected
    
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
        
        # Reset counters
        self.domain_counter = Counter()
        self.path_components = defaultdict(list)
        self.sections = defaultdict(Counter)
        self.topic_words = Counter()
        
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
        
        return {
            'category_counts': dict(category_counter),
            'category_top_domains': category_top_domains,
            'dataframe': df,
            'plot': fig
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

def export_results_to_pdf(results, plots=None):
    """
    Export analysis results to a PDF file with embedded images
    
    Args:
        results: The analysis results dict
        plots: Dictionary of plots/figures to include
        
    Returns:
        bytes: PDF file as bytes
    """
    # Convert markdown to HTML
    md_text = results['report']
    html_text = markdown.markdown(md_text)
    
    # Function to save figure to base64
    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    
    # Get base64 encoded images from plots
    base64_images = {}
    if plots:
        for name, fig in plots.items():
            if fig:
                base64_images[name] = fig_to_base64(fig)
    
    # Add styling to make the PDF look better with embedded images
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Kalium URL Analysis Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.5;
                margin: 40px;
                font-size: 11pt;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                font-size: 20pt;
            }}
            h2 {{
                color: #3498db;
                margin-top: 25px;
                font-size: 16pt;
            }}
            h3 {{
                color: #2980b9;
                font-size: 14pt;
            }}
            h4 {{
                color: #2980b9;
                font-size: 12pt;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
            }}
            th {{
                background-color: #f2f2f2;
                text-align: left;
            }}
            img {{
                max-width: 100%;
                height: auto;
                margin: 15px 0;
                page-break-inside: avoid;
            }}
            .image-container {{
                margin: 20px 0;
                page-break-inside: avoid;
            }}
            .domain-item {{
                margin-bottom: 5px;
            }}
            .category-item {{
                margin-bottom: 5px;
            }}
            .page-break {{
                page-break-before: always;
            }}
        </style>
    </head>
    <body>
        {html_text}
        
        <div class="page-break"></div>
        <h2>Visualisierungen</h2>
    """
    
    # Add embedded images
    plot_titles = {
        'domain_plot': 'Top Domains',
        'category_plot': 'Inhaltskategorien',
        'section_plot': 'Top Sektionen/Bereiche',
        'topic_plot': 'Häufigste Themen',
        'matrix_plot': 'Domain-Kategorie-Matrix'
    }
    
    for name, title in plot_titles.items():
        if name in base64_images:
            html_content += f"""
            <div class="image-container">
                <h3>{title}</h3>
                <img src="data:image/png;base64,{base64_images[name]}" alt="{title}" />
            </div>
            """
    
    html_content += """
    </body>
    </html>
    """
    
    # Create a temporary HTML file
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
        f.write(html_content.encode('utf-8'))
        temp_html = f.name
    
    # Convert HTML to PDF
    try:
        pdf_buffer = io.BytesIO()
        font_config = FontConfiguration()
        HTML(temp_html).write_pdf(pdf_buffer, stylesheets=[], font_config=font_config)
        pdf_buffer.seek(0)
        
        # Clean up temporary file
        os.unlink(temp_html)
        
        return pdf_buffer.getvalue()
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        # Clean up temporary file
        os.unlink(temp_html)
        return None

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
                        'matrix_plot': results['matrix_fig']
                    }
                    
                st.success(f"Analyse abgeschlossen in {results['execution_time']:.2f} Sekunden!")
                st.rerun()

elif input_method == "URLs einfügen":
    st.subheader("URLs direkt einfügen")
    url_text = st.text_area("Füge URLs hier ein (eine URL pro Zeile oder gemischt mit Text):", height=300)
    
    if url_text:
        analyzer = URLAnalyzer()
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
                        'matrix_plot': results['matrix_fig']
                    }
                    
                st.success(f"Analyse abgeschlossen in {results['execution_time']:.2f} Sekunden!")
                st.rerun()

# Display results if available
if 'results' in st.session_state and st.session_state.results:
    results = st.session_state.results
    plots = st.session_state.plots
    
    # Create tabs for different result sections
    tabs = st.tabs(["Überblick", "Domains", "Kategorien", "Sektionen", "Themen", "Bericht"])
    
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
    
    # Download buttons
    st.sidebar.header("Downloads")
    
    # PDF download
    if st.sidebar.button("Report als PDF herunterladen"):
        with st.spinner("Erstelle PDF..."):
            pdf_bytes = export_results_to_pdf(results, plots)
            
            if pdf_bytes:
                st.sidebar.download_button(
                    label="PDF herunterladen",
                    data=pdf_bytes,
                    file_name="kalium_url_analysis.pdf",
                    mime="application/pdf"
                )
    
    # CSV downloads
    if st.sidebar.button("Daten als CSV herunterladen"):
        # Create a zip file with all CSV data
        buffer = io.BytesIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save dataframes to CSV
            results['domain_results']['dataframe'].to_csv(f"{temp_dir}/domains.csv", index=False)
            results['category_results']['dataframe'].to_csv(f"{temp_dir}/categories.csv", index=False)
            results['section_results']['dataframe'].to_csv(f"{temp_dir}/sections.csv", index=False)
            results['topic_results']['dataframe'].to_csv(f"{temp_dir}/topics.csv", index=False)
            
            # Save report as markdown
            with open(f"{temp_dir}/report.md", 'w') as f:
                f.write(results['report'])
            
            # Create zip file
            import zipfile
            with zipfile.ZipFile(buffer, 'w') as zip_file:
                for file in os.listdir(temp_dir):
                    zip_file.write(os.path.join(temp_dir, file), file)
        
        buffer.seek(0)
        st.sidebar.download_button(
            label="CSV-Dateien herunterladen",
            data=buffer,
            file_name="kalium_analysis_data.zip",
            mime="application/zip"
        )

# Footer
st.markdown("---")
st.markdown("Kalium URL Analyzer - v1.0")
