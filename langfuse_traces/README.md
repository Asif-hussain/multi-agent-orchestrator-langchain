# Langfuse Traces Screenshots

This folder contains screenshots from the Langfuse dashboard showing the system's observability features.

## Screenshots Included

### langfuse_dashboard_overview.png
Shows the main traces dashboard with multiple queries processed by the system. You can see both RetrievalQA and ChatOpenAI traces with their inputs and outputs.

### langfuse_trace_hr_remote_work.png
Detailed trace for an HR query about remote work policy. Shows the full RetrievalQA chain including vector store retrieval and document stuffing steps.

### langfuse_trace_orchestrator_classification.png
Shows the orchestrator's classification logic. Includes the reasoning for routing the query to HR department and the model parameters used.

### langfuse_trace_it_vpn_query.png
Trace for an IT support query about VPN password reset. Shows the IT agent's execution path and retrieved documents from the tech documentation.

## How to Capture Your Own Screenshots

1. Run the notebook and process some queries
2. Go to cloud.langfuse.com and navigate to your project
3. Click on "Traces" to see all executed queries
4. Click individual traces to see detailed execution paths
5. Take screenshots as needed

These screenshots help demonstrate the observability features without requiring access to the Langfuse account.
