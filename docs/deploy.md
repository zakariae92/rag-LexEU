# Deploying to one VM

The public demo runs on a single Oracle Cloud *Always Free* VM: `VM.Standard.A1.Flex`, 4 ARM
(Ampere) cores, 24 GB of RAM, Ubuntu 24.04. The stack is [`deploy/compose.prod.yml`](../deploy/compose.prod.yml):
Caddy (HTTPS), the UI, the API, the embedding server, Qdrant, Postgres and Redis. See
[ADR 0010](adr/0010-deployment.md) for why.

```
push to main ─► CI: tests + retrieval and answer gates ─► arm64 images on GHCR ─► deploy job
                                                                                    │ ssh deploy@VM
                                                           /opt/lexeu: docker compose pull && up
```

The VM is set up once, by hand (below); every release after that is the `deploy` job.

## 1. The VM

Create it in the console (Compute > Instances): shape `VM.Standard.A1.Flex` with 4 OCPUs and 24 GB
(the whole Always Free allowance: 3,000 OCPU-hours and 18,000 GB-hours a month), Ubuntu 24.04, a
public IPv4 address, a 100 GB boot volume. In the subnet's security list, allow TCP 80 and 443 from
`0.0.0.0/0`.

Then, over SSH:

```bash
sudo apt update && sudo apt full-upgrade -y
sudo timedatectl set-timezone Europe/Paris

# Oracle's Ubuntu images have their own iptables rules (don't use ufw): open 80 and 443 before
# the final REJECT, and persist them.
sudo iptables -I INPUT 5 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo iptables -I INPUT 5 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo netfilter-persistent save
```

Docker from Docker's apt repository ([docs](https://docs.docker.com/engine/install/ubuntu/)), with
bounded container logs:

```bash
echo '{ "log-driver": "json-file", "log-opts": { "max-size": "10m", "max-file": "3" } }' \
  | sudo tee /etc/docker/daemon.json
sudo systemctl restart docker
```

## 2. A deploy user for CI

CI connects as `deploy`, with its own key, so it can be revoked without touching the admin's
access. (Membership of the `docker` group is equivalent to root on the VM: the key is a production
credential.)

```bash
sudo adduser --disabled-password --gecos "" deploy
sudo usermod -aG docker deploy
sudo mkdir -p /opt/lexeu /home/deploy/.ssh
sudo chown deploy:deploy /opt/lexeu && sudo chmod 2775 /opt/lexeu   # setgid: new files keep the group
sudo usermod -aG deploy ubuntu                                     # the admin can manage it too
```

On your machine, a key pair used only by CI:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/lexeu_deploy -N "" -C "github-actions-deploy"
```

Its public half goes to the VM:

```bash
echo "<content of lexeu_deploy.pub>" | sudo tee /home/deploy/.ssh/authorized_keys
sudo chown -R deploy:deploy /home/deploy/.ssh && sudo chmod 700 /home/deploy/.ssh && sudo chmod 600 /home/deploy/.ssh/authorized_keys
```

In the repository settings (Settings > Secrets and variables > Actions):

| Kind | Name | Value |
|---|---|---|
| secret | `VM_SSH_KEY` | the private key `~/.ssh/lexeu_deploy` |
| variable | `VM_HOST` | the VM's public IP |
| variable | `LEXEU_DOMAIN` | `51-170-130-195.sslip.io` (or a real domain pointing to the VM) |
| variable | `VM_SSH_KNOWN_HOSTS` | output of `ssh-keyscan -t ed25519 <VM IP>` |
| secret | `GEMINI_API_KEY` | so the answer quality gate runs before every deployment |

Check the scanned host key against the VM's own (`ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`
on the VM, `ssh-keygen -lf <(ssh-keyscan -t ed25519 <IP>)` locally): pinning it is what stops a
man in the middle from receiving the deployment.

The deploy job is skipped while `VM_HOST` is unset.

## 3. Images

The `images` job publishes `ghcr.io/zakariae92/rag-lexeu-{api,embeddings,ui}` for linux/arm64,
tagged by commit and `latest`. After the first run, make the three packages public (GitHub >
Packages > each package > Package settings > Change visibility), or `docker login ghcr.io` on the
VM with a token that can only read packages.

## 4. Data

The index is not rebuilt on the VM: the registry (Postgres) and the vectors (a Qdrant snapshot) are
copied from the machine that built them. On that machine:

```bash
# Registry only: no API keys, logged answers or feedback leave the development database.
docker exec lexeu-postgres-1 pg_dump -U lexeu -d lexeu -Fc \
  --exclude-table-data=api_keys --exclude-table-data=answers --exclude-table-data=feedback \
  > lexeu.dump

# The collection the `chunks` alias points to, as a snapshot file.
COLLECTION=$(curl -s http://127.0.0.1:6333/aliases | jq -r '.result.aliases[] | select(.alias_name=="chunks") | .collection_name')
SNAPSHOT=$(curl -s -X POST "http://127.0.0.1:6333/collections/$COLLECTION/snapshots" | jq -r .result.name)
curl -s -o "$COLLECTION.snapshot" "http://127.0.0.1:6333/collections/$COLLECTION/snapshots/$SNAPSHOT"

scp lexeu.dump "$COLLECTION.snapshot" ubuntu@<VM IP>:/opt/lexeu/
```

On the VM, in `/opt/lexeu` (with `compose.prod.yml`, `Caddyfile` and a filled `.env`, from
[`deploy/.env.example`](../deploy/.env.example)):

```bash
docker compose -f compose.prod.yml up -d --wait postgres qdrant

docker compose -f compose.prod.yml exec -T postgres pg_restore -U lexeu -d lexeu --no-owner < lexeu.dump

mkdir -p snapshots && mv chunks_*.snapshot snapshots/
docker run --rm --network lexeu_default curlimages/curl -s -X PUT \
  "http://qdrant:6333/collections/<collection>/snapshots/recover" \
  -H 'Content-Type: application/json' -d '{"location": "file:///qdrant/snapshots/<collection>.snapshot"}'
docker run --rm --network lexeu_default curlimages/curl -s -X POST "http://qdrant:6333/collections/aliases" \
  -H 'Content-Type: application/json' \
  -d '{"actions": [{"create_alias": {"collection_name": "<collection>", "alias_name": "chunks"}}]}'
```

## 5. First start

```bash
docker compose -f compose.prod.yml up -d --wait
docker compose -f compose.prod.yml run --rm api lexeu keys create web-ui --rate-limit 60
# put the printed key in .env as UI_API_KEY, then:
docker compose -f compose.prod.yml up -d --wait
```

`https://<LEXEU_DOMAIN>/` serves the UI, `/docs` the API reference, `/v1/*` the API (API key
required). Caddy obtains the certificate on the first request.

## Operating

| Task | Command (in `/opt/lexeu`) |
|---|---|
| State and health | `docker compose -f compose.prod.yml ps` |
| Logs of a service | `docker compose -f compose.prod.yml logs -f --tail 100 api` |
| Roll back | set `LEXEU_TAG` in `.env` to a previous commit, then `docker compose -f compose.prod.yml up -d --wait` |
| Revoke a key | `docker compose -f compose.prod.yml run --rm api lexeu keys revoke <name>` |
| Back up Postgres | `docker compose -f compose.prod.yml exec -T postgres pg_dump -U lexeu -Fc lexeu > backup.dump` |
