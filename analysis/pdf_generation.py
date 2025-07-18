#!/usr/bin/env python3
"""
PDF Generation Module for Theater Script Conversion.

This module provides functionality to convert conversation CSV files into
theater script format PDFs with metadata and analysis sections. It includes
comprehensive logging and error handling for production use.

Classes:
    TheaterScriptPDFGenerator: Main class for PDF generation

Functions:
    convert_conversation_to_pdf: Convert single CSV to PDF
    batch_convert_conversations: Convert multiple CSV files
    get_formatting_settings: Get default formatting configuration
    setup_logging: Configure logging for the module

Author: AI Assistant
Date: 2025
"""

import json
import logging
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging for the PDF generation module.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        logging.Logger: Configured logger instance

    Example:
        >>> logger = setup_logging("DEBUG")
        >>> logger.info("Starting PDF generation")
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Create console handler if not already exists
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Create formatter with 79-char line limit consideration
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# Initialize logger
logger = setup_logging()


class TheaterScriptPDFGenerator:
    """
    Generate theater script PDFs from conversation data.

    This class handles the conversion of conversation CSV files into
    professionally formatted theater script PDFs with metadata sections,
    conversation content, and analysis data.

    Attributes:
        styles: ReportLab sample styles
        title_style: Custom style for titles
        character_style: Custom style for character names
        dialog_style: Custom style for dialog text
        metadata_style: Custom style for metadata
        section_header_style: Custom style for section headers
        stage_direction_style: Custom style for stage directions

    Example:
        >>> generator = TheaterScriptPDFGenerator()
        >>> generator.generate_pdf("data.csv", "meta.json", "output.pdf")
    """

    def __init__(self):
        """
        Initialize the PDF generator with default styles.

        Sets up ReportLab styles and configures custom styles for
        theater script formatting.
        """
        logger.info("Initializing TheaterScriptPDFGenerator")

        try:
            self.styles = getSampleStyleSheet()
            self.setup_custom_styles()
            logger.debug("Successfully initialized PDF generator styles")
        except Exception as e:
            logger.error(f"Failed to initialize PDF generator: {e}")
            raise

    def setup_custom_styles(self) -> None:
        """
        Configure custom styles for theater script formatting.

        Creates specialized paragraph styles for different elements
        of the theater script including titles, character names,
        dialog, metadata, and stage directions.

        Raises:
            Exception: If style configuration fails
        """
        logger.debug("Setting up custom PDF styles")

        try:
            # Title style - centered, large, prominent
            self.title_style = ParagraphStyle(
                "CustomTitle",
                parent=self.styles["Title"],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER,
            )

            # Character name style - bold, left-aligned
            self.character_style = ParagraphStyle(
                "Character",
                parent=self.styles["Normal"],
                fontSize=12,
                fontName="Helvetica-Bold",
                spaceAfter=6,
                leftIndent=0,
                alignment=TA_LEFT,
            )

            # Dialog style - indented, justified
            self.dialog_style = ParagraphStyle(
                "Dialog",
                parent=self.styles["Normal"],
                fontSize=11,
                leftIndent=20,
                rightIndent=20,
                spaceAfter=12,
                alignment=TA_JUSTIFY,
            )

            # Metadata style - smaller, informational
            self.metadata_style = ParagraphStyle(
                "Metadata",
                parent=self.styles["Normal"],
                fontSize=10,
                fontName="Helvetica",
                spaceAfter=6,
                leftIndent=10,
            )

            # Section header style - centered, prominent
            self.section_header_style = ParagraphStyle(
                "SectionHeader",
                parent=self.styles["Heading2"],
                fontSize=16,
                spaceAfter=15,
                spaceBefore=20,
                alignment=TA_CENTER,
            )

            # Stage direction style - italic, centered, indented
            self.stage_direction_style = ParagraphStyle(
                "StageDirection",
                parent=self.styles["Normal"],
                fontSize=10,
                fontName="Helvetica-Oblique",
                leftIndent=40,
                rightIndent=40,
                spaceAfter=8,
                alignment=TA_CENTER,
            )

            logger.debug("Successfully configured custom styles")

        except Exception as e:
            logger.error(f"Failed to setup custom styles: {e}")
            raise

    def format_character_name(self, role: str) -> str:
        """
        Format character names for theater script conventions.

        Converts role names to uppercase theater script format with
        standardized mappings for common roles.

        Args:
            role: The character role (e.g., 'user', 'assistant', 'system')

        Returns:
            str: Formatted character name in uppercase

        Example:
            >>> generator.format_character_name('user')
            'USER'
            >>> generator.format_character_name('assistant')
            'ASSISTANT'
        """

        return role.upper()

    def wrap_text(self, text: str, max_width: int = 79) -> str:
        """
        Wrap text to specified width while preserving line breaks.

        Processes text to ensure it fits within the specified character
        width limit while maintaining the original line break structure.

        Args:
            text: The text to wrap
            max_width: Maximum characters per line (default: 79)

        Returns:
            str: Wrapped text with preserved line breaks

        Example:
            >>> generator.wrap_text("This is a very long line...", 20)
            'This is a very long\\nline...'
        """
        if not text:
            logger.debug("Empty text provided for wrapping")
            return ""

        logger.debug(f"Wrapping text with max_width={max_width}")

        try:
            lines = text.split("\n")
            wrapped_lines = []

            for line in lines:
                if len(line) <= max_width:
                    wrapped_lines.append(line)
                else:
                    wrapped_lines.extend(textwrap.wrap(line, width=max_width))

            result = "\n".join(wrapped_lines)
            logger.debug(
                f"Successfully wrapped text: {len(lines)} -> "
                f"{len(wrapped_lines)} lines"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to wrap text: {e}")
            return text  # Return original text if wrapping fails

    def create_metadata_section(self, metadata: Dict[str, Any]) -> List:
        """
        Create the metadata section for the PDF.

        Builds a formatted metadata section containing experiment
        information, configuration details, and complete JSON dump.

        Args:
            metadata: Dictionary containing experiment metadata

        Returns:
            List: List of ReportLab story elements for the metadata section

        Raises:
            Exception: If metadata processing fails
        """
        logger.info("Creating metadata section")

        try:
            story = []

            # Title
            story.append(
                Paragraph("EXPERIMENT METADATA", self.section_header_style)
            )
            story.append(Spacer(1, 12))

            # Basic information
            basic_fields = [
                ("timestamp", "Timestamp"),
                ("csv_filename", "CSV File"),
                ("num_iterations", "Number of Iterations"),
                ("total_tokens", "Total Tokens"),
            ]

            for field, label in basic_fields:
                if field in metadata:
                    story.append(
                        Paragraph(
                            f"<b>{label}:</b> {metadata[field]}",
                            self.metadata_style,
                        )
                    )
                    logger.debug(f"Added metadata field: {field}")

            story.append(Spacer(1, 12))

            # Configuration section
            if "config" in metadata:
                logger.debug("Adding configuration section")
                story.append(
                    Paragraph("<b>Configuration:</b>", self.metadata_style)
                )

                config = metadata["config"]
                for key, value in config.items():
                    story.append(
                        Paragraph(
                            f"&nbsp;&nbsp;&nbsp;&nbsp;<b>{key}:</b> {value}",
                            self.metadata_style,
                        )
                    )

            story.append(Spacer(1, 20))

            # Complete JSON dump
            logger.debug("Adding complete JSON metadata")
            story.append(
                Paragraph(
                    "<b>Complete Metadata JSON:</b>", self.metadata_style
                )
            )
            story.append(Spacer(1, 6))

            json_text = json.dumps(metadata, indent=2)
            wrapped_json = self.wrap_text(json_text, max_width=85)

            json_style = ParagraphStyle(
                "JSONStyle",
                parent=self.styles["Normal"],
                fontSize=9,
                fontName="Courier",
                leftIndent=20,
                spaceAfter=6,
            )

            for line in wrapped_json.split("\n"):
                story.append(Paragraph(line, json_style))

            story.append(PageBreak())

            logger.info("Successfully created metadata section")
            return story

        except Exception as e:
            logger.error(f"Failed to create metadata section: {e}")
            raise

    def create_conversation_section(self, df: pd.DataFrame) -> List:
        """
        Create the main conversation section in theater script format.

        Converts the conversation DataFrame into a formatted theater
        script with character names, dialog, and stage directions.

        Args:
            df: DataFrame containing conversation data with columns:
                - iteration: Conversation iteration number
                - role: Character role (user, assistant, system)
                - response: The actual dialog text
                - timestamp: When the response was generated

        Returns:
            List: List of ReportLab story elements for the conversation

        Raises:
            Exception: If conversation processing fails
        """
        logger.info("Creating conversation section")

        try:
            story = []

            # Title
            story.append(Paragraph("THE CONVERSATION", self.title_style))
            story.append(Spacer(1, 30))

            # Sort by iteration to ensure proper order
            df_sorted = df.sort_values("iteration")
            logger.debug(f"Processing {len(df_sorted)} conversation entries")

            current_scene = 1

            for idx, row in df_sorted.iterrows():
                # Add scene breaks every 20 iterations
                if row["iteration"] % 20 == 0 and row["iteration"] > 0:
                    logger.debug(
                        f"Adding scene break at iteration "
                        f"{row['iteration']}"
                    )
                    story.append(Spacer(1, 20))
                    story.append(
                        Paragraph(
                            f"--- SCENE {current_scene} ---",
                            self.stage_direction_style,
                        )
                    )
                    story.append(Spacer(1, 20))
                    current_scene += 1

                # Character name
                character_name = self.format_character_name(row["role"])
                story.append(
                    Paragraph(f"{character_name}:", self.character_style)
                )

                # Dialog
                # Determine which content column exists and has data
                content_col = None
                for col in ["content", "response"]:
                    if col in df.columns and pd.notna(row[col]):
                        content_col = col
                        break

                # Get content from determined column
                # or empty string if none found
                response_text = str(row[content_col]) if content_col else ""
                wrapped_response = self.wrap_text(response_text)

                # Split into paragraphs for better formatting
                paragraphs = wrapped_response.split("\n\n")
                for para in paragraphs:
                    if para.strip():
                        story.append(
                            Paragraph(para.strip(), self.dialog_style)
                        )

                # Add iteration info as stage direction
                stage_info = (
                    f"(Iteration {row['iteration']}, " f"{row['timestamp']})"
                )
                story.append(Paragraph(stage_info, self.stage_direction_style))

                story.append(Spacer(1, 15))

            logger.info(
                f"Successfully created conversation section with "
                f"{len(df_sorted)} entries"
            )
            return story

        except Exception as e:
            logger.error(f"Failed to create conversation section: {e}")
            raise

    def create_analysis_section(
        self, df: pd.DataFrame, include_table: bool = True
    ) -> List:
        """
        Create the analysis section with metrics and statistics.

        Generates a comprehensive analysis section containing summary
        statistics, role distribution, metrics over time, and sample data.

        Args:
            df: DataFrame containing conversation data and metrics

        Returns:
            List: List of ReportLab story elements for the analysis section

        Raises:
            Exception: If analysis processing fails
        """
        logger.info("Creating analysis section")

        try:
            story = []

            story.append(PageBreak())
            story.append(Paragraph("ANALYSIS DATA", self.section_header_style))
            story.append(Spacer(1, 20))

            # Summary statistics
            logger.debug("Generating summary statistics")
            story.append(
                Paragraph("<b>Summary Statistics</b>", self.metadata_style)
            )
            story.append(Spacer(1, 8))

            # Basic stats
            total_iterations = len(df)
            story.append(
                Paragraph(
                    f"Total Iterations: {total_iterations}",
                    self.metadata_style,
                )
            )

            # Word count statistics
            if "word_count" in df.columns:
                total_words = df["word_count"].sum()
                avg_words = df["word_count"].mean()
                story.append(
                    Paragraph(
                        f"Total Words: {total_words}", self.metadata_style
                    )
                )
                story.append(
                    Paragraph(
                        f"Average Words per Response: {avg_words:.2f}",
                        self.metadata_style,
                    )
                )
                logger.debug(
                    f"Added word count statistics: "
                    f"{total_words} total, {avg_words:.2f} avg"
                )

            # Unique word statistics
            if "unique_word_count" in df.columns:
                avg_unique_words = df["unique_word_count"].mean()
                story.append(
                    Paragraph(
                        f"Average Unique Words: {avg_unique_words:.2f}",
                        self.metadata_style,
                    )
                )
                logger.debug(
                    f"Added unique word statistics: "
                    f"{avg_unique_words:.2f} avg"
                )

            # Role distribution
            logger.debug("Generating role distribution")
            role_counts = df["role"].value_counts()
            story.append(Spacer(1, 12))
            story.append(
                Paragraph("<b>Role Distribution</b>", self.metadata_style)
            )
            for role, count in role_counts.items():
                story.append(
                    Paragraph(
                        f"{role}: {count} responses", self.metadata_style
                    )
                )

            story.append(Spacer(1, 20))

            # Metrics summary
            logger.debug("Generating metrics summary")
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                story.append(
                    Paragraph("<b>Metrics Summary</b>", self.metadata_style)
                )
                story.append(Spacer(1, 8))

                for col in numeric_columns:
                    if col != "iteration" and df[col].notna().any():
                        mean_val = df[col].mean()
                        std_val = df[col].std()
                        story.append(
                            Paragraph(
                                f"{col}: Mean={mean_val:.3f}, "
                                f"Std={std_val:.3f}",
                                self.metadata_style,
                            )
                        )

            story.append(Spacer(1, 20))

            # Complete data table
            logger.debug("Creating complete data table")
            story.append(
                Paragraph("<b>Complete Data</b>", self.metadata_style)
            )
            story.append(Spacer(1, 8))

            # Create table
            table_data = self._create_sample_table(df)

            # Set column widths to make table more compact
            # Each column should be able to fit at most 8 characters
            num_columns = len(table_data[0]) if table_data else 0
            col_widths = [0.8 * inch] * num_columns  # 0.8 inch per column

            table = Table(table_data, colWidths=col_widths)
            table.setStyle(self._get_table_style())

            if include_table:
                story.append(table)

            logger.info("Successfully created analysis section")
            return story

        except Exception as e:
            logger.error(f"Failed to create analysis section: {e}")
            raise

    def _create_sample_table(  # noqa: C901
        self, df: pd.DataFrame
    ) -> List[List[str]]:
        """
        Create table data for data display.

        Args:
            df: DataFrame to convert to table format

        Returns:
            List[List[str]]: Table data with headers and rows
        """
        logger.debug("Creating data table")

        table_data = []

        # Parse analysis column if it exists
        analysis_columns = []
        if "analysis" in df.columns:
            try:
                # Get all unique analysis metrics from the first few rows
                analysis_metrics = set()
                for _, row in df.head(10).iterrows():
                    if pd.notna(row["analysis"]):
                        try:
                            analysis_data = eval(row["analysis"])
                            if isinstance(analysis_data, dict):
                                analysis_metrics.update(analysis_data.keys())
                        except Exception as e:
                            logger.error(f"Error parsing analysis data: {e}")
                            continue

                analysis_columns = sorted(list(analysis_metrics))
                logger.debug(f"Found analysis columns: {analysis_columns}")
            except Exception as e:
                logger.warning(f"Failed to parse analysis column: {e}")
                analysis_columns = []

        # Build headers from all available columns
        headers = ["Iteration", "Role", "Timestamp"]

        # Add analysis columns if available
        if analysis_columns:
            headers.extend(
                [col.replace("_", " ").title() for col in analysis_columns]
            )

        # Add other relevant columns
        other_columns = ["agent_id", "fetcher_config"]
        for col in other_columns:
            if col in df.columns:
                headers.append(col.replace("_", " ").title())

        table_data.append(headers)

        # Data rows
        for _, row in df.iterrows():
            # Handle iteration (short)
            iteration_str = str(row.get("iteration", "N/A"))

            # Handle role (wrap if needed)
            role_str = str(row.get("role", "N/A"))
            if len(role_str) > 8:
                role_str = self.wrap_text(role_str, max_width=8)

            # Handle timestamp (wrap if needed)
            timestamp_str = str(row.get("timestamp", "N/A"))
            if len(timestamp_str) > 8:
                # Extract just the time part if it's a full timestamp
                if " " in timestamp_str:
                    timestamp_str = timestamp_str.split(" ")[1][:8]
                else:
                    timestamp_str = timestamp_str[:8]

            table_row = [iteration_str, role_str, timestamp_str]

            # Add analysis data if available
            if analysis_columns and "analysis" in df.columns:
                analysis_data = {}
                if pd.notna(row["analysis"]):
                    try:
                        analysis_data = eval(row["analysis"])
                        if not isinstance(analysis_data, dict):
                            analysis_data = {}
                    except Exception as e:
                        logger.error(f"Error parsing analysis data: {e}")
                        analysis_data = {}

                for col in analysis_columns:
                    value = analysis_data.get(col, "N/A")
                    if isinstance(value, float) and value is not None:
                        table_row.append(f"{value:.3f}")
                    else:
                        table_row.append(
                            str(value) if value is not None else "N/A"
                        )

            # Add other columns
            for col in other_columns:
                if col in df.columns:
                    value = row.get(col, "N/A")
                    value_str = str(value)

                    # Handle agent_id and other long values with wrapping
                    if col == "agent_id" and len(value_str) > 8:
                        value_str = self.wrap_text(value_str, max_width=8)
                    elif col == "fetcher_config" and len(value_str) > 8:
                        # For fetcher_config, just show the fetcher type
                        if "fetcher" in value_str:
                            try:
                                import re

                                fetcher_match = re.search(
                                    r"'fetcher': <FetcherType\.(\w+):",
                                    value_str,
                                )
                                if fetcher_match:
                                    value_str = fetcher_match.group(1)[:8]
                                else:
                                    value_str = value_str[:8]
                            except Exception as e:
                                logger.error(
                                    f"Error parsing fetcher config: {e}"
                                )
                                value_str = value_str[:8]
                        else:
                            value_str = value_str[:8]
                    elif len(value_str) > 8:
                        value_str = value_str[:8]

                    table_row.append(value_str)

            table_data.append(table_row)

        logger.debug(
            f"Created complete table with {len(table_data)} rows and "
            f"{len(headers)} columns"
        )
        return table_data

    def _get_table_style(self) -> TableStyle:
        """
        Get the standard table style for data tables.

        Returns:
            TableStyle: Configured table style
        """
        return TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("FONTSIZE", (0, 1), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
            ]
        )

    def generate_pdf(
        self,
        csv_file: str,
        json_file: str,
        output_file: str,
        include_table: bool = False,
    ) -> str:
        """
        Generate the complete PDF from CSV and JSON files.

        Main method that orchestrates the entire PDF generation process,
        reading data files, creating sections, and building the final PDF.

        Args:
            csv_file: Path to the CSV file containing conversation data
            json_file: Path to the JSON file containing metadata
            output_file: Path where the generated PDF will be saved

        Returns:
            str: Path to the generated PDF file

        Raises:
            FileNotFoundError: If input files don't exist
            Exception: If PDF generation fails

        Example:
            >>> generator = TheaterScriptPDFGenerator()
            >>> pdf_path = generator.generate_pdf(
            ...     "conversation.csv",
            ...     "metadata.json",
            ...     "output.pdf"
            ... )
        """
        logger.info(f"Starting PDF generation: {csv_file} -> {output_file}")

        try:
            # Validate input files
            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"CSV file not found: {csv_file}")
            if not os.path.exists(json_file):
                raise FileNotFoundError(f"JSON file not found: {json_file}")

            # Read data
            logger.debug(f"Reading CSV file: {csv_file}")
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(df)} conversation entries")

            logger.debug(f"Reading JSON file: {json_file}")
            with open(json_file, "r") as f:
                metadata = json.load(f)
            logger.info("Loaded metadata successfully")

            # Create PDF document
            logger.debug(f"Creating PDF document: {output_file}")
            doc = SimpleDocTemplate(
                output_file,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72,
            )

            # Build story
            logger.debug("Building PDF story sections")
            story = []

            # Add sections
            story.extend(self.create_metadata_section(metadata))
            story.extend(self.create_conversation_section(df))
            story.extend(
                self.create_analysis_section(df, include_table=include_table)
            )

            # Build PDF
            logger.debug("Building final PDF")
            doc.build(story)

            logger.info(f"PDF generated successfully: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            raise


def convert_conversation_to_pdf(
    csv_filename: str,
    json_filename: Optional[str] = None,
    output_dir: str = "data/experiment_pdfs",
) -> str:
    """
    Convert a conversation CSV file to a theater script PDF.

    High-level function that handles the conversion of a single CSV file
    to a theater script PDF, with automatic metadata file detection and
    temporary metadata creation if needed.

    Args:
        csv_filename: Path to the CSV file containing conversation data
        json_filename: Path to the JSON metadata file (optional, auto-detected)
        output_dir: Directory to save the generated PDF
        (default: data/experiment_pdfs)

    Returns:
        str: Path to the generated PDF file

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        Exception: If conversion fails

    Example:
        >>> pdf_path = convert_conversation_to_pdf(
        ...     "conversation.csv",
        ...     "metadata.json",
        ...     "output/"
        ... )
    """
    logger.info(f"Converting conversation to PDF: {csv_filename}")

    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Created output directory: {output_dir}")

        # Auto-detect JSON file if not specified
        if json_filename is None:
            base_name = os.path.splitext(csv_filename)[0]
            json_filename = f"{base_name}_meta.json"
            logger.debug(f"Auto-detected JSON filename: {json_filename}")

        # Check if files exist
        if not os.path.exists(csv_filename):
            raise FileNotFoundError(f"CSV file not found: {csv_filename}")

        temp_json_created = False
        if not os.path.exists(json_filename):
            logger.warning(f"JSON file not found: {json_filename}")
            logger.info("Creating minimal metadata...")

            # Create minimal metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "csv_filename": os.path.basename(csv_filename),
                "config": {"note": "No metadata file found"},
            }

            # Create temporary JSON file
            base_name = os.path.splitext(csv_filename)[0]
            json_filename = f"{base_name}_temp_meta.json"

            with open(json_filename, "w") as f:
                json.dump(metadata, f, indent=2)

            temp_json_created = True
            logger.debug(f"Created temporary JSON file: {json_filename}")

        # Generate output filename
        base_name = os.path.splitext(os.path.basename(csv_filename))[0]
        output_file = os.path.join(
            output_dir, f"{base_name}_theater_script.pdf"
        )

        # Generate PDF
        generator = TheaterScriptPDFGenerator()
        result = generator.generate_pdf(
            csv_filename, json_filename, output_file
        )

        # Clean up temporary file if created
        if temp_json_created:
            os.remove(json_filename)
            logger.debug(f"Cleaned up temporary file: {json_filename}")

        logger.info(f"Successfully converted to PDF: {result}")
        return result

    except Exception as e:
        logger.error(f"Failed to convert conversation to PDF: {e}")
        raise


def batch_convert_conversations(
    data_dir: str, output_dir: str = "data/experiment_pdfs"
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Convert all CSV files in a directory to theater script PDFs.

    Processes multiple CSV files in a directory, converting each to a
    theater script PDF with comprehensive error handling and reporting.

    Args:
        data_dir: Directory containing CSV files to convert
        output_dir: Directory to save generated PDFs
        (default: data/experiment_pdfs)

    Returns:
        Tuple[List[str], List[Tuple[str, str]]]:
            - List of successfully converted PDF files
            - List of (filename, error_message) tuples for failed conversions

    Example:
        >>> successful, failed = batch_convert_conversations("data/")
        >>> print(f"Converted {len(successful)} files, {len(failed)} failed")
    """
    logger.info(f"Starting batch conversion from directory: {data_dir}")

    try:
        data_path = Path(data_dir)
        csv_files = list(data_path.rglob("*.csv"))

        if not csv_files:
            logger.warning(f"No CSV files found in {data_dir}")
            return [], []

        logger.info(f"Found {len(csv_files)} CSV files to convert")

        successful_conversions = []
        failed_conversions = []

        for csv_file in csv_files:
            try:
                logger.debug(f"Converting: {csv_file.name}")
                pdf_file = convert_conversation_to_pdf(
                    str(csv_file), output_dir=output_dir
                )
                successful_conversions.append(pdf_file)
                logger.info(f"✓ Converted: {csv_file.name}")

            except Exception as e:
                error_msg = str(e)
                failed_conversions.append((csv_file.name, error_msg))
                logger.error(f"✗ Failed: {csv_file.name} - {error_msg}")

        # Final summary
        logger.info("Batch conversion complete!")
        logger.info(f"Successful: {len(successful_conversions)}")
        logger.info(f"Failed: {len(failed_conversions)}")

        if failed_conversions:
            logger.warning("Failed conversions:")
            for filename, error in failed_conversions:
                logger.warning(f"  - {filename}: {error}")

        return successful_conversions, failed_conversions

    except Exception as e:
        logger.error(f"Batch conversion failed: {e}")
        raise


def get_formatting_settings() -> Dict[str, Any]:
    """
    Get the default formatting settings for PDF generation.

    Returns a comprehensive dictionary of formatting settings that can be
    used to understand or modify the PDF generation behavior.

    Returns:
        Dict[str, Any]: Dictionary containing all formatting settings

    Example:
        >>> settings = get_formatting_settings()
        >>> print(f"Max text width: {settings['text_wrapping']['max_width']}")
    """
    logger.debug("Retrieving formatting settings")

    settings = {
        "page_size": "letter",
        "margins": {"top": 72, "bottom": 72, "left": 72, "right": 72},
        "text_wrapping": {"max_width": 79, "preserve_line_breaks": True},
        "scene_breaks": {
            "interval": 20,  # Every 20 iterations
            "enabled": True,
        },
        "fonts": {
            "title": {"name": "Title", "size": 24},
            "character": {"name": "Helvetica-Bold", "size": 12},
            "dialog": {"name": "Helvetica", "size": 11},
            "metadata": {"name": "Helvetica", "size": 10},
            "stage_direction": {"name": "Helvetica-Oblique", "size": 10},
            "json": {"name": "Courier", "size": 9},
        },
        "indentation": {
            "dialog_left": 20,
            "dialog_right": 20,
            "stage_direction_left": 40,
            "stage_direction_right": 40,
            "metadata_left": 10,
            "json_left": 20,
        },
    }

    logger.debug("Retrieved formatting settings successfully")
    return settings


if __name__ == "__main__":
    """
    Main execution block for testing and demonstration.

    Provides example usage of the PDF generation functionality with
    proper error handling and logging.
    """
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("PDF Generation Script Started")
    logger.info("=" * 50)

    # Example file paths
    csv_file = "data/test_raven_ollama/drift_experiment_20250528_140953.csv"
    json_file = (
        "data/test_raven_ollama/drift_experiment_20250528_140953_meta.json"
    )

    # Test single file conversion
    if os.path.exists(csv_file):
        logger.info(f"Converting example file: {csv_file}")
        try:
            pdf_file = convert_conversation_to_pdf(csv_file, json_file)
            logger.info(f"Successfully created: {pdf_file}")
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
    else:
        logger.warning(f"Example file not found: {csv_file}")

    # Display formatting settings
    logger.info("\nFormatting Settings:")
    logger.info("-" * 20)
    settings = get_formatting_settings()
    for key, value in settings.items():
        logger.info(f"  {key}: {value}")

    logger.info("PDF Generation Script Completed")
