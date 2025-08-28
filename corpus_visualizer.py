#!/usr/bin/env python3
"""
Corpus Analysis Visualization Tool
Creates charts, networks, and interactive visualizations from corpus analysis results
"""

import json
import logging
from pathlib import Path
import argparse
from collections import Counter, defaultdict

# Data analysis libraries
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Network analysis
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# Interactive plotting
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

class CorpusVisualizer:
    def __init__(self, results_dir: Path, output_dir: Path):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load corpus summary
        summary_file = self.results_dir / "corpus_summary.json"
        if not summary_file.exists():
            raise FileNotFoundError(f"Corpus summary not found: {summary_file}")
        
        with open(summary_file) as f:
            self.summary = json.load(f)
        
        # Load individual results
        self.individual_results = self._load_individual_results()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Setup plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def _load_individual_results(self) -> list[dict]:
        """Load all individual analysis results"""
        results = []
        results_dir = self.results_dir / "individual_results"
        
        if results_dir.exists():
            for json_file in results_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                        result = json.load(f)
                        result['_filename'] = json_file.stem
                        results.append(result)
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to load {json_file}")
        
        return results

    def create_domain_analysis(self):
        """Create domain distribution visualizations"""
        stats = self.summary.get("corpus_statistics", {})
        domains = stats.get("domains", {})
        
        if not domains:
            self.logger.warning("No domain data available")
            return
        
        # Create bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Top domains bar chart
        top_domains = dict(list(domains.items())[:15])
        ax1.barh(list(top_domains.keys()), list(top_domains.values()))
        ax1.set_title('Top 15 Research Domains')
        ax1.set_xlabel('Number of Papers')
        
        # Domain distribution pie chart for top 10
        top_10_domains = dict(list(domains.items())[:10])
        other_count = sum(domains.values()) - sum(top_10_domains.values())
        if other_count > 0:
            top_10_domains['Others'] = other_count
        
        ax2.pie(top_10_domains.values(), labels=top_10_domains.keys(), autopct='%1.1f%%')
        ax2.set_title('Research Domain Distribution (Top 10)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "domain_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create interactive plotly version if available
        if HAS_PLOTLY:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Domain Distribution', 'Top Domains'),
                specs=[[{"type": "pie"}, {"type": "bar"}]]
            )
            
            # Pie chart
            fig.add_trace(
                go.Pie(labels=list(top_10_domains.keys()), values=list(top_10_domains.values())),
                row=1, col=1
            )
            
            # Bar chart
            fig.add_trace(
                go.Bar(x=list(top_domains.values()), y=list(top_domains.keys()), orientation='h'),
                row=1, col=2
            )
            
            fig.update_layout(title_text="Research Domain Analysis", showlegend=False)
            fig.write_html(str(self.output_dir / "domain_analysis_interactive.html"))

    def create_technique_analysis(self):
        """Analyze and visualize research techniques"""
        stats = self.summary.get("corpus_statistics", {})
        techniques = stats.get("techniques", {})
        
        if not techniques:
            self.logger.warning("No technique data available")
            return
        
        # Create word cloud
        if techniques:
            wordcloud = WordCloud(
                width=1200, height=600, 
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate_from_frequencies(techniques)
            
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Research Techniques Word Cloud', fontsize=16, pad=20)
            plt.savefig(self.output_dir / "techniques_wordcloud.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Top techniques bar chart
        top_techniques = dict(list(techniques.items())[:20])
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_techniques)), list(top_techniques.values()))
        plt.yticks(range(len(top_techniques)), list(top_techniques.keys()))
        plt.xlabel('Number of Papers')
        plt.title('Top 20 Research Techniques')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.output_dir / "top_techniques.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_innovation_network(self):
        """Create network analysis of innovations and techniques"""
        if not HAS_NETWORKX:
            self.logger.warning("NetworkX not available, skipping network analysis")
            return
        
        G = nx.Graph()
        
        # Build network from individual results
        for result in self.individual_results:
            if 'innovations' not in result or 'methodology' not in result:
                continue
            
            innovations = result.get('innovations', {}).get('innovations_claimed', [])
            techniques = result.get('methodology', {}).get('techniques_used', [])
            domain = result.get('content_analysis', {}).get('research_domain', 'Unknown')
            
            # Add domain as central node
            G.add_node(f"Domain:{domain}", node_type='domain', size=10)
            
            # Connect innovations to techniques
            for innovation in innovations[:3]:  # Limit to top 3
                innovation_node = f"Innovation:{innovation}"
                G.add_node(innovation_node, node_type='innovation', size=5)
                G.add_edge(f"Domain:{domain}", innovation_node)
                
                for technique in techniques[:3]:  # Limit to top 3
                    technique_node = f"Technique:{technique}"
                    G.add_node(technique_node, node_type='technique', size=3)
                    G.add_edge(innovation_node, technique_node)
        
        # Remove isolated nodes and very low-degree nodes
        nodes_to_remove = [node for node, degree in G.degree() if degree < 2]
        G.remove_nodes_from(nodes_to_remove)
        
        if len(G.nodes()) == 0:
            self.logger.warning("No network data available")
            return
        
        # Create network visualization
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Color nodes by type
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            if node.startswith('Domain:'):
                node_colors.append('red')
                node_sizes.append(300)
            elif node.startswith('Innovation:'):
                node_colors.append('blue')
                node_sizes.append(200)
            else:  # Technique
                node_colors.append('green')
                node_sizes.append(100)
        
        nx.draw(G, pos, 
               node_color=node_colors, 
               node_size=node_sizes,
               with_labels=False,  # Too many labels would be cluttered
               alpha=0.7,
               edge_color='gray',
               width=0.5)
        
        plt.title('Innovation-Technique Network', fontsize=16, pad=20)
        plt.savefig(self.output_dir / "innovation_network.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save network statistics
        network_stats = {
            'nodes': len(G.nodes()),
            'edges': len(G.edges()),
            'density': nx.density(G),
            'clustering_coefficient': nx.average_clustering(G) if len(G.nodes()) > 0 else 0,
            'top_nodes_by_degree': [
                {'node': node, 'degree': degree} 
                for node, degree in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]
            ]
        }
        
        with open(self.output_dir / "network_stats.json", 'w') as f:
            json.dump(network_stats, f, indent=2)

    def create_temporal_analysis(self):
        """Analyze trends over time if publication years are available"""
        years_data = []
        domains_by_year = defaultdict(list)
        
        for result in self.individual_results:
            year = result.get('metadata', {}).get('publication_year')
            domain = result.get('content_analysis', {}).get('research_domain', 'Unknown')
            
            if year and isinstance(year, int) and 1990 <= year <= 2025:
                years_data.append(year)
                domains_by_year[year].append(domain)
        
        if not years_data:
            self.logger.warning("No temporal data available")
            return
        
        # Create temporal visualization
        year_counts = Counter(years_data)
        sorted_years = sorted(year_counts.items())
        
        plt.figure(figsize=(12, 6))
        years, counts = zip(*sorted_years)
        plt.plot(years, counts, marker='o', linewidth=2, markersize=6)
        plt.xlabel('Publication Year')
        plt.ylabel('Number of Papers')
        plt.title('Publication Trends Over Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "temporal_trends.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Domain evolution over time
        if len(set(years_data)) > 3:  # Only if we have multiple years
            domain_evolution = {}
            all_domains = set()
            
            for year in sorted(set(years_data)):
                year_domains = Counter(domains_by_year[year])
                domain_evolution[year] = year_domains
                all_domains.update(year_domains.keys())
            
            # Plot top domains over time
            top_domains = list(self.summary.get("corpus_statistics", {}).get("domains", {}).keys())[:5]
            
            plt.figure(figsize=(12, 6))
            for domain in top_domains:
                domain_counts = [domain_evolution.get(year, {}).get(domain, 0) for year in sorted(set(years_data))]
                plt.plot(sorted(set(years_data)), domain_counts, marker='o', label=domain, linewidth=2)
            
            plt.xlabel('Publication Year')
            plt.ylabel('Number of Papers')
            plt.title('Research Domain Evolution Over Time')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / "domain_evolution.png", dpi=300, bbox_inches='tight')
            plt.close()

    def create_complexity_analysis(self):
        """Analyze technical complexity and innovation scores"""
        complexity_scores = []
        innovation_scores = []
        impact_scores = []
        
        for result in self.individual_results:
            relevance = result.get('relevance_scores', {})
            if isinstance(relevance, dict):
                if 'technical_complexity' in relevance:
                    complexity_scores.append(relevance['technical_complexity'])
                if 'innovation_score' in relevance:
                    innovation_scores.append(relevance['innovation_score'])
                if 'practical_impact' in relevance:
                    impact_scores.append(relevance['practical_impact'])
        
        if not any([complexity_scores, innovation_scores, impact_scores]):
            self.logger.warning("No scoring data available")
            return
        
        # Create multi-plot analysis
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Complexity distribution
        if complexity_scores:
            axes[0, 0].hist(complexity_scores, bins=range(1, 7), alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Technical Complexity Distribution')
            axes[0, 0].set_xlabel('Complexity Score (1-5)')
            axes[0, 0].set_ylabel('Number of Papers')
        
        # Innovation distribution  
        if innovation_scores:
            axes[0, 1].hist(innovation_scores, bins=range(1, 7), alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Innovation Score Distribution')
            axes[0, 1].set_xlabel('Innovation Score (1-5)')
            axes[0, 1].set_ylabel('Number of Papers')
        
        # Impact distribution
        if impact_scores:
            axes[1, 0].hist(impact_scores, bins=range(1, 7), alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Practical Impact Distribution')
            axes[1, 0].set_xlabel('Impact Score (1-5)')
            axes[1, 0].set_ylabel('Number of Papers')
        
        # Correlation plot
        if complexity_scores and innovation_scores:
            min_len = min(len(complexity_scores), len(innovation_scores))
            axes[1, 1].scatter(complexity_scores[:min_len], innovation_scores[:min_len], alpha=0.6)
            axes[1, 1].set_xlabel('Technical Complexity')
            axes[1, 1].set_ylabel('Innovation Score')
            axes[1, 1].set_title('Complexity vs Innovation')
            
            # Add correlation coefficient
            if min_len > 1:
                corr = np.corrcoef(complexity_scores[:min_len], innovation_scores[:min_len])[0, 1]
                axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                              transform=axes[1, 1].transAxes, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "complexity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_summary_dashboard(self):
        """Create a comprehensive summary dashboard"""
        if not HAS_PLOTLY:
            self.logger.warning("Plotly not available, skipping interactive dashboard")
            return
        
        stats = self.summary.get("corpus_statistics", {})
        proc_stats = self.summary.get("processing_summary", {})
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Processing Summary', 'Top Domains', 'Top Techniques', 'Research Types'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Processing summary pie
        fig.add_trace(go.Pie(
            labels=['Successful', 'Failed'],
            values=[proc_stats.get('successful', 0), proc_stats.get('failed', 0)],
            name="Processing"
        ), row=1, col=1)
        
        # Top domains bar
        domains = stats.get("domains", {})
        top_domains = dict(list(domains.items())[:10])
        fig.add_trace(go.Bar(
            x=list(top_domains.keys()),
            y=list(top_domains.values()),
            name="Domains"
        ), row=1, col=2)
        
        # Top techniques bar
        techniques = stats.get("techniques", {})
        top_techniques = dict(list(techniques.items())[:15])
        fig.add_trace(go.Bar(
            x=list(top_techniques.values()),
            y=list(top_techniques.keys()),
            orientation='h',
            name="Techniques"
        ), row=2, col=1)
        
        # Research types pie
        research_types = stats.get("research_types", {})
        fig.add_trace(go.Pie(
            labels=list(research_types.keys()),
            values=list(research_types.values()),
            name="Research Types"
        ), row=2, col=2)
        
        fig.update_layout(
            title_text="Corpus Analysis Dashboard",
            showlegend=False,
            height=800
        )
        
        # Save interactive dashboard
        fig.write_html(str(self.output_dir / "dashboard.html"))

    def export_data_for_analysis(self):
        """Export processed data in various formats for further analysis"""
        # Create structured DataFrame
        data_records = []
        
        for result in self.individual_results:
            record = {
                'filename': result.get('_filename', ''),
                'title': result.get('metadata', {}).get('title', ''),
                'year': result.get('metadata', {}).get('publication_year', None),
                'domain': result.get('content_analysis', {}).get('research_domain', ''),
                'main_topic': result.get('content_analysis', {}).get('main_topic', ''),
                'research_type': result.get('methodology', {}).get('research_type', ''),
                'techniques': ', '.join(result.get('methodology', {}).get('techniques_used', [])),
                'innovations': ', '.join(result.get('innovations', {}).get('innovations_claimed', [])),
                'complexity_score': result.get('relevance_scores', {}).get('technical_complexity', None),
                'innovation_score': result.get('relevance_scores', {}).get('innovation_score', None),
                'impact_score': result.get('relevance_scores', {}).get('practical_impact', None),
                'keywords': ', '.join(result.get('keywords', [])),
            }
            data_records.append(record)
        
        # Save as CSV
        df = pd.DataFrame(data_records)
        df.to_csv(self.output_dir / "corpus_data.csv", index=False)
        
        # Save as Excel with multiple sheets
        with pd.ExcelWriter(self.output_dir / "corpus_analysis.xlsx", engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Full_Data', index=False)
            
            # Domain summary
            domain_summary = df.groupby('domain').agg({
                'title': 'count',
                'complexity_score': 'mean',
                'innovation_score': 'mean',
                'impact_score': 'mean'
            }).round(2)
            domain_summary.columns = ['Paper_Count', 'Avg_Complexity', 'Avg_Innovation', 'Avg_Impact']
            domain_summary.to_excel(writer, sheet_name='Domain_Summary')
            
            # Year summary (if available)
            year_data = df[df['year'].notna()]
            if not year_data.empty:
                year_summary = year_data.groupby('year').agg({
                    'title': 'count',
                    'complexity_score': 'mean',
                    'innovation_score': 'mean'
                }).round(2)
                year_summary.columns = ['Paper_Count', 'Avg_Complexity', 'Avg_Innovation']
                year_summary.to_excel(writer, sheet_name='Year_Summary')

    def generate_all_visualizations(self):
        """Generate all visualizations and analyses"""
        self.logger.info("Creating domain analysis...")
        self.create_domain_analysis()
        
        self.logger.info("Creating technique analysis...")
        self.create_technique_analysis()
        
        self.logger.info("Creating innovation network...")
        self.create_innovation_network()
        
        self.logger.info("Creating temporal analysis...")
        self.create_temporal_analysis()
        
        self.logger.info("Creating complexity analysis...")
        self.create_complexity_analysis()
        
        self.logger.info("Creating summary dashboard...")
        self.create_summary_dashboard()
        
        self.logger.info("Exporting data...")
        self.export_data_for_analysis()
        
        # Create summary HTML report
        self._create_html_report()

    def _create_html_report(self):
        """Create comprehensive HTML report with all visualizations"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Corpus Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .section {{ margin: 30px 0; }}
        .stats {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .image {{ text-align: center; margin: 20px 0; }}
        img {{ max-width: 100%; height: auto; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Academic Paper Corpus Analysis</h1>
        <p>Generated by gptme Corpus Analyzer</p>
    </div>
    
    <div class="section stats">
        <h2>Processing Summary</h2>
        <p><strong>Total Papers:</strong> {self.summary.get('processing_summary', {}).get('total_papers', 'N/A')}</p>
        <p><strong>Successfully Processed:</strong> {self.summary.get('processing_summary', {}).get('successful', 'N/A')}</p>
        <p><strong>Success Rate:</strong> {self.summary.get('processing_summary', {}).get('success_rate', 0)*100:.1f}%</p>
    </div>
    
    <div class="section">
        <h2>Domain Analysis</h2>
        <div class="image">
            <img src="domain_analysis.png" alt="Domain Analysis">
        </div>
    </div>
    
    <div class="section">
        <h2>Research Techniques</h2>
        <div class="grid">
            <div class="image">
                <img src="techniques_wordcloud.png" alt="Techniques Word Cloud">
            </div>
            <div class="image">
                <img src="top_techniques.png" alt="Top Techniques">
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Innovation Networks</h2>
        <div class="image">
            <img src="innovation_network.png" alt="Innovation Network">
        </div>
    </div>
    
    <div class="section">
        <h2>Temporal Analysis</h2>
        <div class="image">
            <img src="temporal_trends.png" alt="Temporal Trends">
        </div>
    </div>
    
    <div class="section">
        <h2>Complexity Analysis</h2>
        <div class="image">
            <img src="complexity_analysis.png" alt="Complexity Analysis">
        </div>
    </div>
    
    <div class="section">
        <h2>Interactive Dashboard</h2>
        <p><a href="dashboard.html">Open Interactive Dashboard</a></p>
    </div>
    
    <div class="section">
        <h2>Data Exports</h2>
        <ul>
            <li><a href="corpus_data.csv">Raw Data (CSV)</a></li>
            <li><a href="corpus_analysis.xlsx">Analysis Workbook (Excel)</a></li>
            <li><a href="network_stats.json">Network Statistics (JSON)</a></li>
        </ul>
    </div>
</body>
</html>
"""
        
        with open(self.output_dir / "report.html", 'w') as f:
            f.write(html_content)


def main():
    parser = argparse.ArgumentParser(description="Visualize corpus analysis results")
    parser.add_argument("results_dir", help="Directory containing corpus analysis results")
    parser.add_argument("-o", "--output", default="corpus_visualizations", 
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    visualizer = CorpusVisualizer(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output)
    )
    
    print("Generating visualizations...")
    visualizer.generate_all_visualizations()
    print(f"Visualizations saved to: {args.output}")
    print(f"Open {args.output}/report.html to view the full report")


if __name__ == "__main__":
    main()
