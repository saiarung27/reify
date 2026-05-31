import { useState } from 'react';
import { ReifyClient, Memory, Transition } from '@reify/sdk';

const client = new ReifyClient('/api');

function App() {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [transitions, setTransitions] = useState<Transition[]>([]);
  const [knowledge, setKnowledge] = useState<Memory[]>([]);
  const [identityId, setIdentityId] = useState('');
  const [query, setQuery] = useState('');
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'memories' | 'transitions' | 'knowledge'>('memories');

  const handleRecall = async () => {
    if (!query) return;
    setLoading(true);
    try {
      const response = await client.recall(query);
      setMemories(response.results);
    } catch (error) {
      console.error('Recall failed:', error);
    }
    setLoading(false);
  };

  const handleRemember = async () => {
    if (!text) return;
    setLoading(true);
    try {
      await client.remember(text);
      setText('');
      await handleRecall();
    } catch (error) {
      console.error('Remember failed:', error);
    }
    setLoading(false);
  };

  const handleWhatChanged = async () => {
    setLoading(true);
    try {
      const response = await client.whatChanged(7);
      setTransitions(response.transitions);
    } catch (error) {
      console.error('WhatChanged failed:', error);
    }
    setLoading(false);
  };

  const handleWhatDoYouKnow = async () => {
    if (!identityId) return;
    setLoading(true);
    try {
      const response = await client.whatDoYouKnow(identityId);
      setKnowledge(response.memories);
    } catch (error) {
      console.error('WhatDoYouKnow failed:', error);
    }
    setLoading(false);
  };

  return (
    <div className="container">
      <div className="header">
        <h1>Reify Dashboard</h1>
        <p>Browse claims, transitions, and memories</p>
      </div>

      <div className="card">
        <h2>Remember</h2>
        <div className="input-group">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter content to remember..."
            rows={3}
          />
          <button onClick={handleRemember} disabled={loading}>
            Remember
          </button>
        </div>
      </div>

      <div className="card">
        <h2>Recall</h2>
        <div className="input-group">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search query..."
          />
          <button onClick={handleRecall} disabled={loading}>
            Recall
          </button>
        </div>
      </div>

      <div className="card">
        <div className="tabs">
          <button
            className={activeTab === 'memories' ? 'active' : ''}
            onClick={() => setActiveTab('memories')}
          >
            Memories ({memories.length})
          </button>
          <button
            className={activeTab === 'transitions' ? 'active' : ''}
            onClick={() => { setActiveTab('transitions'); handleWhatChanged(); }}
          >
            Transitions ({transitions.length})
          </button>
          <button
            className={activeTab === 'knowledge' ? 'active' : ''}
            onClick={() => setActiveTab('knowledge')}
          >
            What Do You Know
          </button>
        </div>

        {activeTab === 'memories' ? (
          memories.length === 0 ? (
            <p>No memories found</p>
          ) : (
            <ul className="memory-list">
              {memories.map((memory) => (
                <li key={memory.id} className="memory-item">
                  <div>{memory.content}</div>
                  <div className="memory-meta">
                    {memory.network} • {new Date(memory.created_at).toLocaleString()}
                  </div>
                </li>
              ))}
            </ul>
          )
        ) : activeTab === 'transitions' ? (
          transitions.length === 0 ? (
            <p>No transitions found</p>
          ) : (
            <ul className="memory-list">
              {transitions.map((transition, idx) => (
                <li key={idx} className="memory-item">
                  <div>Memory: {transition.memory_id}</div>
                  <div className="memory-meta">
                    {transition.from_state} → {transition.to_state} • {new Date(transition.at).toLocaleString()}
                  </div>
                </li>
              ))}
            </ul>
          )
        ) : (
          <div>
            <div className="input-group">
              <input
                value={identityId}
                onChange={(e) => setIdentityId(e.target.value)}
                placeholder="Enter identity ID..."
              />
              <button onClick={handleWhatDoYouKnow} disabled={loading}>
                Query
              </button>
            </div>
            {knowledge.length === 0 ? (
              <p>No knowledge found for this identity</p>
            ) : (
              <ul className="memory-list">
                {knowledge.map((memory) => (
                  <li key={memory.id} className="memory-item">
                    <div>{memory.content}</div>
                    <div className="memory-meta">
                      {memory.network} • {new Date(memory.created_at).toLocaleString()}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
