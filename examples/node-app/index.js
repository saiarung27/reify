/**
 * Example Node.js application using Reify REST API
 *
 * This example demonstrates how to use the Reify memory engine
 * via its REST API from a Node.js application.
 */

import fetch from 'node-fetch';

const API_BASE = 'http://localhost:8003';

async function remember(text) {
  const response = await fetch(`${API_BASE}/remember`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) {
    throw new Error(`Remember failed: ${response.statusText}`);
  }
  return response.json();
}

async function recall(query, limit = 20) {
  const response = await fetch(`${API_BASE}/recall`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, limit }),
  });
  if (!response.ok) {
    throw new Error(`Recall failed: ${response.statusText}`);
  }
  return response.json();
}

async function whatChanged(days = 7) {
  const response = await fetch(`${API_BASE}/what-changed?days=${days}`);
  if (!response.ok) {
    throw new Error(`WhatChanged failed: ${response.statusText}`);
  }
  return response.json();
}

async function whatDoYouKnow(identityId) {
  const response = await fetch(`${API_BASE}/what-do-you-know?identity_id=${encodeURIComponent(identityId)}`);
  if (!response.ok) {
    throw new Error(`WhatDoYouKnow failed: ${response.statusText}`);
  }
  return response.json();
}

async function main() {
  console.log('Reify Node.js Example\n');

  // Remember some facts
  console.log('Remembering facts...');
  await remember('I work at Stripe as a Staff Engineer.');
  await remember('I live in San Francisco with my dog Mochi.');
  await remember('I prefer TypeScript over JavaScript for new projects.');
  console.log('Facts remembered.\n');

  // Recall facts
  console.log('Recalling: Where do I work?');
  const workResult = await recall('Where do I work?');
  console.log(`Results: ${workResult.count}`);
  workResult.results.forEach((r, i) => {
    console.log(`  ${i + 1}. ${r.content} (${r.network})`);
  });
  console.log();

  console.log('Recalling: What pet do I have?');
  const petResult = await recall('What pet do I have?');
  console.log(`Results: ${petResult.count}`);
  petResult.results.forEach((r, i) => {
    console.log(`  ${i + 1}. ${r.content} (${r.network})`);
  });
  console.log();

  // Check for changes
  console.log('Checking for changes in the last 7 days...');
  const changes = await whatChanged(7);
  console.log(`Transitions: ${changes.count}`);
  changes.transitions.forEach((t, i) => {
    console.log(`  ${i + 1}. ${t.memory_id}: ${t.from_state} → ${t.to_state} at ${t.at}`);
  });
  console.log();

  // What do you know (using default identity)
  console.log('What do you know about me?');
  const knowledge = await whatDoYouKnow('default');
  console.log(`Memories: ${knowledge.count}`);
  knowledge.memories.forEach((m, i) => {
    console.log(`  ${i + 1}. ${m.content} (${m.network})`);
  });
  console.log();

  console.log('Example complete.');
}

main().catch(console.error);
