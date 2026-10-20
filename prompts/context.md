# About this environment

<!--
A short, generic description of the shape of the infrastructure the tools reach.
Do NOT put a real host inventory, real IP addresses, or real endpoint URLs here
if this file will be published — keep those in a private copy of the prompt
directory. The tool schemas already tell the model which systems it can query;
this file is only for orientation.
-->

- A single-node hypervisor runs a handful of Linux containers and VMs.
- Monitoring is a metrics database plus a dashboard/alerting front end on a
  separate small host.
- DNS for the network is a filtering resolver on that same monitoring host.
- An uptime monitor writes a plain availability log that one of the tools reads.

The exact host names, addresses, and URLs are whatever the configured MCP
servers report — always use those, not anything in this file.
