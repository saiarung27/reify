/**
 * Reify TypeScript SDK
 * 
 * Client library for the Reify memory engine REST API.
 */

export interface Memory {
  id: string;
  content: string;
  created_at: string;
  network: string;
}

export interface RecallResponse {
  query: string;
  results: Memory[];
  count: number;
}

export interface RememberResponse {
  committed: number;
  memory_ids: string[];
}

export interface Transition {
  memory_id: string;
  from_state: string;
  to_state: string;
  at: string;
}

export class ReifyClient {
  private baseUrl: string;

  constructor(baseUrl: string = "http://localhost:8001") {
    this.baseUrl = baseUrl;
  }

  async remember(text: string, identityId?: string): Promise<RememberResponse> {
    const response = await fetch(`${this.baseUrl}/remember`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, identity_id: identityId }),
    });
    if (!response.ok) throw new Error(`Remember failed: ${response.statusText}`);
    return response.json();
  }

  async recall(query: string, identityId?: string, limit = 20): Promise<RecallResponse> {
    const response = await fetch(`${this.baseUrl}/recall`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, identity_id: identityId, limit }),
    });
    if (!response.ok) throw new Error(`Recall failed: ${response.statusText}`);
    return response.json();
  }

  async forget(memoryId?: string, tag?: string, identityId?: string): Promise<{ deleted: number }> {
    const response = await fetch(`${this.baseUrl}/forget`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ memory_id: memoryId, tag, identity_id: identityId }),
    });
    if (!response.ok) throw new Error(`Forget failed: ${response.statusText}`);
    return response.json();
  }

  async whatChanged(days = 7): Promise<{ transitions: Transition[]; count: number }> {
    const response = await fetch(`${this.baseUrl}/what-changed?days=${days}`);
    if (!response.ok) throw new Error(`WhatChanged failed: ${response.statusText}`);
    return response.json();
  }

  async whatDoYouKnow(identityId: string): Promise<{ identity_id: string; memories: Memory[]; count: number }> {
    const response = await fetch(`${this.baseUrl}/what-do-you-know?identity_id=${encodeURIComponent(identityId)}`);
    if (!response.ok) throw new Error(`WhatDoYouKnow failed: ${response.statusText}`);
    return response.json();
  }
}
