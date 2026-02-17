import React from 'react';
import { FiCheckCircle, FiInfo } from 'react-icons/fi';

function ResultsDisplay({ result }) {
  // Guard against undefined result
  if (!result || !result.predicted_class) {
    return (
      <div className="results-display">
        <div className="results-header">
          <FiCheckCircle size={32} color="#EF4444" />
          <h3>Error</h3>
        </div>
        <p style={{ color: '#EF4444' }}>Unable to display results. Please try again.</p>
      </div>
    );
  }

  const getTypeColor = (typeName) => {
    if (!typeName) return '#666';
    switch (typeName) {
      case 'type-1':
        return '#FF6B6B';
      case 'type-2':
        return '#4ECDC4';
      case 'type-3':
        return '#45B7D1';
      default:
        return '#666';
    }
  };

  const getTypeIcon = (typeName) => {
    if (!typeName) return '🏛️';
    const icons = {
      'type-1': '🎭',
      'type-2': '🔷',
      'type-3': '▭'
    };
    return icons[typeName] || '🏛️';
  };

  return (
    <div className="results-display">
      <div className="results-header">
        <FiCheckCircle size={32} color="#10B981" />
        <h3>Classification Results</h3>
      </div>

      <div className="main-prediction">
        <div 
          className="prediction-box"
          style={{ borderLeftColor: getTypeColor(result.predicted_class) }}
        >
          <div className="prediction-header">
            <span className="type-icon">{getTypeIcon(result.predicted_class)}</span>
            <div>
              <h4>Predicted Type</h4>
              <p className="type-name">{result.predicted_class.toUpperCase()}</p>
            </div>
          </div>
          
          <div className="confidence-section">
            <p className="confidence-label">Confidence Level</p>
            <div className="confidence-bar">
              <div 
                className="confidence-fill"
                style={{ width: `${result.confidence}%` }}
              ></div>
            </div>
            <p className="confidence-value">{result.confidence.toFixed(2)}%</p>
          </div>

          {result.class_descriptions && (
            <div className="description">
              <h5>{result.class_descriptions.name}</h5>
              <p>{result.class_descriptions.definition}</p>
              {result.class_descriptions.characteristics && (
                <div className="characteristics">
                  <p><strong>Key Characteristics:</strong></p>
                  <ul>
                    {result.class_descriptions.characteristics.map((char, idx) => (
                      <li key={idx}>{char}</li>
                    ))}
                  </ul>
                </div>
              )}
              {result.class_descriptions.example && (
                <p><strong>Example:</strong> {result.class_descriptions.example}</p>
              )}
              {result.class_descriptions.patronage && (
                <p><strong>Patronage:</strong> {result.class_descriptions.patronage}</p>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="probabilities">
        <h4>All Predictions</h4>
        <div className="probability-list">
          {Object.entries(result.all_probabilities).map(([className, probability]) => (
            <div key={className} className="probability-item">
              <span className="class-label">{className.toUpperCase()}</span>
              <div className="probability-bar">
                <div 
                  className="probability-fill"
                  style={{ width: `${probability}%` }}
                ></div>
              </div>
              <span className="probability-value">{probability.toFixed(2)}%</span>
            </div>
          ))}
        </div>
      </div>

      {result.method === 'ensemble' && result.models_used && (
        <div className="ensemble-info">
          <FiInfo size={20} />
          <p>
            This result was generated using {result.num_models || result.models_used.length} different models:
            {' '}{result.models_used.join(', ').toUpperCase()}
          </p>
        </div>
      )}

      <div className="timestamp">
        <small>Prediction made at: {new Date(result.timestamp).toLocaleString()}</small>
      </div>
    </div>
  );
}

export default ResultsDisplay;
