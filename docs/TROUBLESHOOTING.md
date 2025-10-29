# Ring Pipeline Troubleshooting Guide

## Issues Fixed

### ✅ Issue 1: Single Node Timeout
**Problem**: Single node was sending tensors to itself but timing out.
**Fix**: Added special case for `world_size == 1` to process locally without network messages.

### ✅ Issue 2: Second Node Not Detected
**Problem**: Topology not updating when second node joins.
**Fixes**: 
1. Added topology broadcast when LLM service starts
2. Added topology_update message handler in `main.py`
3. Increased wait time for peer discovery (3 seconds)

## Testing with 2 Nodes

### Terminal 1 (Head Node)
```bash
cd HyperCluster-v0.1
python main.py start --ring

# Wait for document creation, copy the ticket
# IMPORTANT: Don't start LLM yet!
```

### Terminal 2 (Worker Node)
```bash
cd HyperCluster-v0.1
python main.py start --ring --bootstrap-ticket "<PASTE_TICKET_HERE>"

# You should see: "Joined document with ID: ..."
```

### Back to Terminal 1 - Start LLM on BOTH nodes
```bash
# In Terminal 1:
> llm start Qwen/Qwen2.5-0.5B-Instruct

# Watch for:
# "Broadcasting topology update to discover peers..."
# "Found 2 nodes in topology"  ← This is key!
# "Rank: 0/2"  ← Should show 2 nodes
```

### Terminal 2 - Start LLM
```bash
# In Terminal 2:
> llm start Qwen/Qwen2.5-0.5B-Instruct

# Watch for:
# "Found 2 nodes in topology"
# "Rank: 1/2"  ← Should show rank 1
```

### Send Query (Terminal 1 only)
```bash
# Only on Terminal 1 (head node):
> llm query "What is a neural network?"

# You should see layers distributed!
```

## What You Should See

### Correct 2-Node Setup

**Terminal 1 (Rank 0):**
```
🔍 Initializing ring with 2 nodes in topology
   Node 0: 11cb71871ac731bc... - 16.0 GB
   Node 1: d79f0f8e5819e45f... - 16.0 GB

🌍 Full Cluster Distribution:
👉 Rank 0 (HEAD): Layers   0- 11 (12 layers)
   Rank 1 (WORK-1): Layers  12- 23 (12 layers)
```

**Terminal 2 (Rank 1):**
```
🌍 Full Cluster Distribution:
   Rank 0 (HEAD): Layers   0- 11 (12 layers)
👉 Rank 1 (WORK-1): Layers  12- 23 (12 layers)
```

### During Inference

**Terminal 1:**
```
⚙️  Processing on Rank 0
   Layers: 0 → 11 (12 layers)
   📤 Forwarding to Rank 1 (d79f0f8e5819e45f...)
```

**Terminal 2:**
```
📥 RECEIVED TENSOR
   From: 11cb71871ac731bc...
⚙️  Processing on Rank 1
   Layers: 12 → 23 (12 layers)
   📤 Sending final result to HEAD
```

## Common Issues

### Issue: "Found 1 nodes in topology"
**Cause**: Second node hasn't broadcast its topology yet.
**Fix**: 
1. Make sure BOTH nodes run `llm start` command
2. Wait 3-5 seconds between starting nodes
3. Check that both nodes are in the same document (check doc ID)

### Issue: "Timeout waiting for ring completion"
**Cause**: Messages not being received between nodes.
**Fix**:
1. Check firewall isn't blocking ports
2. Verify both nodes are connected to same Iroh document
3. Try `peers` command to see if nodes see each other

### Issue: "Rank 0: 16.0 GB (100.0%) → 24 layers"
**Cause**: Only one node detected.
**Fix**: Start LLM service on BOTH nodes, not just one!

### Issue: No response to query
**Cause**: Query might be sent before ring is ready.
**Fix**: Wait for "Ring pipeline ready" message before sending query.

## Debug Commands

### Check Peers
```bash
> peers
# Should show other nodes
```

### Check Topology
Add this to your node to see topology:
```python
# After llm start:
print(f"Topology nodes: {len(node.topology.all_nodes())}")
for node_id, cap in node.topology.all_nodes():
    print(f"  - {node_id[:16]}: {cap.memory} GB")
```

## Expected Timeline

1. **Node 1 starts**: ~2 seconds
2. **Node 2 joins**: ~2 seconds  
3. **Node 1 starts LLM**: ~5-8 seconds (model loading + topology broadcast)
4. **Node 2 starts LLM**: ~5-8 seconds
5. **Query sent**: Should see response within 30 seconds

## Verification Checklist

- [ ] Both nodes show same document ID
- [ ] `peers` command shows other node
- [ ] "Found 2 nodes in topology" appears on both
- [ ] Ring initialization shows "Rank: 0/2" and "Rank: 1/2"
- [ ] Layer distribution is split (not 100% on one node)
- [ ] Query produces response without timeout

## Performance Expectations

### Single Node (Ring Mode)
- Should work without timeout
- Processes all layers locally
- ~500-800ms per token (Qwen-0.5B)

### Two Nodes (True Ring)
- Layer distribution: 50/50 or proportional to memory
- ~600-1000ms per token (includes network overhead)
- Bottleneck: Usually network transfer time

## Still Having Issues?

1. **Check logs**: Look for ERROR or WARNING messages
2. **Restart clean**: Exit both nodes, start fresh
3. **Single node first**: Test with `--ring` on one node only
4. **Network test**: Try `text hello` to verify Iroh messaging works

## Advanced: Force Specific Layer Distribution

You can manually set layers if auto-detection fails:
```python
# In node.py or llm_service.py, hardcode for testing:
n_layer_window = [12, 12]  # 12 layers per node
```

But this should be automatic once topology is working!

