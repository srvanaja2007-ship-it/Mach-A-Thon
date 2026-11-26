# Orbit-to-Economy Portal

## Deploying on Render

The repository-root `render.yaml` defines free Render web services for the portal,
UrbanPulse, ClaimPulse, and LEO Sim. The Eco_Act dashboard is served as a static
path by the portal. The portal uses each service's public `onrender.com` URL.

To deploy, create a Render Blueprint from this repository and select `render.yaml`.
Free web services may sleep while idle; if a module needs a paid plan to build or
run reliably, review the required plan and cost before upgrading it.

## Run locally

Run `python portal.py` from this directory. The portal launches the local modules
when their links are opened.
